import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import argparse
import tensorboardX
import logging
from losses import sliced_score_estimation_vr
from losses import dsm_score_estimation
from scorenet import ResScore, UNetScore

from torch.utils.data import Dataset

class RepeatingDataset(Dataset):
    def __init__(self, base_tensor, target_len):
        self.base_tensor = base_tensor
        self.base_len = base_tensor.shape[0]
        self.target_len = target_len

    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):
        return (self.base_tensor[idx % self.base_len],)

    #def __getitem__(self, idx):
    #    return self.base_tensor[idx % self.base_len]

def build_path(args):
    '''
    Build the path to save results of training. 
    General pattern is: architecture name, data name, noise level, etc 
    '''
    dir_name = os.path.join(args.dir_name, args.arch_name, args.data_name, 
                          f"{args.noise_level_range[0]}to{args.noise_level_range[1]}")
    
    if args.RF is not None: 
        dir_name = dir_name + f'_RF_{args.RF}x{args.RF}'

    if args.set_size is not None: 
        dir_name = dir_name + f'_set_size_{args.set_size}'
        
    if args.swap:
        dir_name = dir_name + '_swapped'

    return dir_name

def prep_data_swap(train_coeffs, args, save=True):
    '''
    Prepare and potentially swap training and test sets,
    ensuring equal splits and full dataset utilization
    '''
    # Calculate midpoint for even split
    midpoint = len(train_coeffs) // 2
    
    # Override set_size if it's explicitly provided in args
    if args.set_size is None or args.set_size > midpoint:
        args.set_size = midpoint
        logging.info(f"Set size adjusted to {args.set_size} to ensure equal splits")
    
    if args.swap is False:
        # First half for training
        train_set = train_coeffs[:midpoint]
        # Second half for testing
        test_set = train_coeffs[midpoint:]
    else:
        # Second half for training
        train_set = train_coeffs[midpoint:]
        # First half for testing
        test_set = train_coeffs[:midpoint]
    
    # If set_size is smaller than the available half, subsample
    if args.set_size < midpoint:
        train_set = train_set[:args.set_size]
        test_set = test_set[:args.set_size]
    
    # For evaluation, we may want to limit test set size to avoid memory issues
    if args.limit_test_size:
        test_set = test_set[:args.batch_size]
    
    logging.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")
    
    if save: 
        torch.save(train_set, os.path.join(args.dir_name, 'train_set.pt'))
        torch.save(test_set, os.path.join(args.dir_name, 'test_set.pt'))
        
    return train_set, test_set

def repeat_images(train_set, args, N_total): 
    '''
    Repeat images to reach desired total dataset size
    '''
    n = int(N_total/args.set_size)
    train_set = torch.tile(train_set, (n, 1, 1, 1))
    return train_set

class CelebFaceScoreNetRunner:
    def __init__(self, args):
        self.args = args
        
        # Set device
        self.args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup noise conditioning parameters
        self.setup_noise_levels()
        
        # Load data
        self.load_and_prepare_data()
        
        # Build model
        self.build_model()
        
        # Create directory for saving results
        if not os.path.exists(self.args.dir_name):
            os.makedirs(self.args.dir_name)
            
        # Set up tensorboard
        tb_path = os.path.join(self.args.run_dir, 'tensorboard', self.args.experiment_name)
        if os.path.exists(tb_path):
            import shutil
            shutil.rmtree(tb_path)
        self.tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        
    def setup_noise_levels(self):
        """
        Create geometric sequence of noise levels for NCSN training
        """
        L = 10  # Number of noise levels
        sigma_start = 1.0
        sigma_end = 0.01
        
        # Create geometric sequence
        ratio = (sigma_end / sigma_start) ** (1 / (L - 1))
        self.sigmas = torch.tensor([sigma_start * (ratio ** i) for i in range(L)], 
                                  device=self.args.device)
        
        self.logger.info(f"Noise levels: {self.sigmas.cpu().numpy()}")
        
        # Set lambda values for weighted loss (λ(σ) = σ²)
        self.lambdas = self.sigmas ** 2
    
    def get_optimizer(self):
        '''
        Create optimizer based on configuration
        '''
        if self.args.optimizer == 'Adam':
            return optim.Adam(self.score_model.parameters(), lr=self.args.lr, 
                             weight_decay=self.args.weight_decay, betas=(self.args.beta1, 0.999))
        elif self.args.optimizer == 'RMSProp':
            return optim.RMSprop(self.score_model.parameters(), lr=self.args.lr, 
                                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            return optim.SGD(self.score_model.parameters(), lr=self.args.lr, momentum=0.9)
        else:
            raise NotImplementedError(f'Optimizer {self.args.optimizer} not implemented.')
    
    def load_and_prepare_data(self):
        '''
        Load and prepare training and test data
        '''
        self.args.data_path = os.path.join(self.args.data_root_path, self.args.data_name)
        
        # Load raw data
        train_data = torch.load(os.path.join(self.args.data_path, 'train80x80_no_repeats.pt'))
        self.logger.info(f'All data shape: {train_data.shape}')
        
        # Calculate midpoint for even split - will be used in prep_data_swap
        self.total_samples = train_data.shape[0]
        self.midpoint = self.total_samples // 2
        self.logger.info(f"Total samples: {self.total_samples}, Midpoint: {self.midpoint}")
        
        # Adjust set size if it's too large
        if self.midpoint < self.args.set_size: 
            self.args.set_size = self.midpoint
            self.logger.info(f"Set size adjusted to {self.args.set_size} to ensure equal splits")
        
        # Build the directory path
        self.args.dir_name = build_path(self.args)
        if not os.path.exists(self.args.dir_name):
            os.makedirs(self.args.dir_name)
        
        # Select a sub-set and swap if needed
        self.train_set, self.test_set = prep_data_swap(train_data, self.args, save=True)
        self.logger.info(f'Train set shape: {self.train_set.shape}, Test set shape: {self.test_set.shape}')
        
        # Set number of channels
        self.args.num_channels = self.train_set.shape[1]
        
        # For debug mode or repeat training images
        #if self.args.debug:
        #    self.train_set = self.train_set[0:self.args.batch_size]
        #    self.test_set = self.test_set[0:self.args.batch_size]
        #    self.args.num_epochs = 5
        #else:
            # Repeat train images to match desired total size
        #    self.train_set = repeat_images(self.train_set, self.args, N_total=250000)
        
        self.logger.info(f'Final train set shape: {self.train_set.shape}, Test set shape: {self.test_set.shape}')
        
        # Create data loaders
        #self.train_dataset = TensorDataset(self.train_set)
        self.train_dataset = RepeatingDataset(self.train_set, 250000)
        self.test_dataset = TensorDataset(self.test_set)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, 
                                      shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, 
                                     shuffle=False, num_workers=4)
    
    def build_model(self):

        class Config:
            class Data:
                pass
            class Model:
                pass
        
        config = Config()
        config.data = Config.Data()
        config.model = Config.Model()
    
    # Set data attributes
        config.data.channels = self.args.num_channels
        config.data.image_size = 80  # Default to 80 (or set based on your dataset)
        config.data.logit_transform = self.args.logit_transform
    
    # Set model attributes
        config.model.ngf = getattr(self.args, 'ngf', 64)  # Default to 64 if not specified

        config.model.nef = getattr(self.args, 'nef', 64)
        config.model.ndf = getattr(self.args, 'ndf', 64)
    
        # Store original number of channels
        original_channels = config.data.channels

        config.data.out_channels = original_channels  # Always define it
        if self.args.noise_conditioning:
            config.data.channels = config.data.channels + 1
        
        if self.args.arch_name == 'ResScore':
            self.score_model = ResScore(config).to(self.args.device)
        elif self.args.arch_name == 'UNetScore':
            self.score_model = UNetScore(config).to(self.args.device)
        else:
            raise NotImplementedError(f'Architecture {self.args.arch_name} not implemented')
    
        # Restore original number of channels
        self.args.num_channels = original_channels
        
        # Initialize model weights
        if self.args.init_type == 'normal':
            def init_func(m):
                classname = m.__class__.__name__
                if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    nn.init.normal_(m.weight.data, 0.0, self.args.init_gain)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias.data, 0.0)
                elif classname.find('BatchNorm2d') != -1:
                    nn.init.normal_(m.weight.data, 1.0, self.args.init_gain)
                    nn.init.constant_(m.bias.data, 0.0)
            
            self.score_model.apply(init_func)
        
        if torch.cuda.is_available() and self.args.ngpu > 1:
            self.score_model = nn.DataParallel(self.score_model)
        
        self.logger.info(f'Number of parameters: {sum(p.numel() for p in self.score_model.parameters() if p.requires_grad)}')
        
        # Get optimizer
        self.optimizer = self.get_optimizer()
        
        # Load checkpoint if resuming training

        if self.args.ckpt_path:                        # <- NEW
            ckpt_file = self.args.ckpt_path
        elif self.args.resume_training:
            ckpt_file = os.path.join(self.args.log_dir, 'checkpoint.pth')
        else:
            ckpt_file = ''

        if ckpt_file and os.path.exists(ckpt_file):
            states = torch.load(ckpt_file, map_location=self.args.device)
            self.score_model.load_state_dict(states[0])
            self.optimizer.load_state_dict(states[1])
            self.logger.info(f"✓ resumed from {ckpt_file}")
    # optional: restore scheduler / step counter if you store them
        else:
            if ckpt_file:
                self.logger.warning(f"⚠ checkpoint {ckpt_file} not found — training from scratch")

    def cat_sigma_channel(self, x, sigma):
        """
        Helper to concatenate a scalar sigma channel to x.
        x shape: [B, C, H, W]
        returns shape: [B, C+1, H, W]
        """
        B, C, H, W = x.shape
        sigma_map = torch.ones((B, 1, H, W), device=x.device) * sigma
        return torch.cat([x, sigma_map], dim=1)
    
    def train(self):
        '''
        Run the training process with noise conditioning
        '''
        step = 0
        
        for epoch in range(self.args.num_epochs):
            for i, (X,) in enumerate(self.train_loader):
                step += 1
                
                X = X.to(self.args.device)
                if self.args.logit_transform:
                    X = self.logit_transform(X)
                
                # Combined loss across all noise levels
                total_loss = 0.0
                
                # Sample a batch of noise levels
                if self.args.random_noise_level:
                    # Randomly select a noise level for each batch
                    sigma_idx = torch.randint(0, len(self.sigmas), (1,)).item()
                    sigma = self.sigmas[sigma_idx]
                    lambda_weight = self.lambdas[sigma_idx]
                    
                    if self.args.training_algo == 'ssm':
                        X_noisy = X + torch.randn_like(X) * sigma
                        #loss, *_ = sliced_score_estimation_vr(self.score_model, X_noisy.detach(), n_particles=1)
                        loss, loss1, loss2 = sliced_score_estimation_vr(self.score_model, X_noisy.detach(), n_particles=1)
                        #print(f"[TRAIN step {step}] total: {loss.item():.4f}, loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}")
                    elif self.args.training_algo == 'dsm':
                        # If noise_conditioning, do the DSM inline
                        if self.args.noise_conditioning:
                            perturbed = X + torch.randn_like(X) * sigma
                            X_cond = self.cat_sigma_channel(perturbed, sigma)
                            # denoising target
                            target = -(1.0/(sigma**2)) * (perturbed - X)
                            target_flat = target.view(target.shape[0], -1)
                            
                            scores = self.score_model(X_cond)
                            scores_flat = scores.view(scores.shape[0], -1)
                            loss = 0.5 * (scores_flat - target_flat).pow(2).sum(dim=1).mean()
                        else:
                            # old approach
                            loss = dsm_score_estimation(self.score_model, X, sigma=sigma)
                    
                    
                    # Apply lambda weighting as per the paper
                    weighted_loss = lambda_weight * loss
                    total_loss = weighted_loss
                    
                    # Log individual sigma and its loss
                    self.tb_logger.add_scalar(f'sigma_{sigma_idx}_loss', loss.item(), global_step=step)
                    
                else:
                    # Use all noise levels for each batch
                    for idx, sigma in enumerate(self.sigmas):
                        lambda_weight = self.lambdas[idx]
                        
                        if self.args.training_algo == 'ssm':
                            X_noisy = X + torch.randn_like(X) * sigma
                            #loss, *_ = sliced_score_estimation_vr(self.score_model, X_noisy.detach(), n_particles=1)
                            loss, loss1, loss2 = sliced_score_estimation_vr(self.score_model, X_noisy.detach(), n_particles=1)
                           # print(f"[TRAIN step {step}] total: {loss.item():.4f}, loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}")
                        elif self.args.training_algo == 'dsm':
                            if self.args.noise_conditioning:
                                perturbed = X + torch.randn_like(X) * sigma
                                X_cond = self.cat_sigma_channel(perturbed, sigma)
                                target = -(1.0/(sigma**2)) * (perturbed - X)
                                target_flat = target.view(target.shape[0], -1)
                                scores = self.score_model(X_cond)
                                scores_flat = scores.view(scores.shape[0], -1)
                                loss = 0.5 * (scores_flat - target_flat).pow(2).sum(dim=1).mean()
                            else:
                                loss = dsm_score_estimation(self.score_model, X, sigma=sigma)
                        
                        
                        # Apply lambda weighting as per the paper
                        weighted_loss = lambda_weight * loss
                        total_loss += weighted_loss / len(self.sigmas)  # Average across all noise levels
                        
                        # Log individual sigma and its loss
                        self.tb_logger.add_scalar(f'sigma_{idx}_loss', loss.item(), global_step=step)
                
                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Logging
                self.tb_logger.add_scalar('total_loss', total_loss.item(), global_step=step)
                
                if step % 10 == 0:
                    self.logger.info(f"Epoch: {epoch}, Step: {step}, Total Loss: {total_loss.item()}")
                
                # Evaluation on test set
                if step % 100 == 0:
                    self.score_model.eval()
                    #with torch.no_grad():
                    test_iter = iter(self.test_loader)
                    test_X, = next(test_iter)
                    test_X = test_X.to(self.args.device)
                        
                    if self.args.logit_transform:
                        test_X = self.logit_transform(test_X)
                        
                        # Evaluate at multiple noise levels
                    test_total_loss = 0.0
                    for idx, sigma in enumerate(self.sigmas):
                        lambda_weight = self.lambdas[idx]
                            
                        if self.args.training_algo == 'ssm':
                            test_X_noisy = test_X + torch.randn_like(test_X) * sigma
                            #test_loss, *_ = sliced_score_estimation_vr(self.score_model, test_X_noisy, n_particles=1)
                            test_loss, test_loss1, test_loss2 = sliced_score_estimation_vr(self.score_model, test_X_noisy, n_particles=1)
                            print(f"[TEST step {step}] total: {test_loss.item():.4f}, loss1: {test_loss1.item():.4f}, loss2: {test_loss2.item():.4f}")
                        elif self.args.training_algo == 'dsm':
                            with torch.no_grad():
                                if self.args.noise_conditioning:
                                    perturbed_t = test_X + torch.randn_like(test_X) * sigma
                                    test_X_cond = self.cat_sigma_channel(perturbed_t, sigma)
                                    target_t = -(1.0/(sigma**2)) * (perturbed_t - test_X)
                                    target_t_flat = target_t.view(target_t.shape[0], -1)
                                    scores_t = self.score_model(test_X_cond)
                                    scores_t_flat = scores_t.view(scores_t.shape[0], -1)
                                    test_loss = 0.5 * (scores_t_flat - target_t_flat).pow(2).sum(dim=1).mean()
                                else:
                                    test_loss = dsm_score_estimation(self.score_model, test_X, sigma=sigma)
                            
                        weighted_test_loss = lambda_weight * test_loss
                        test_total_loss += weighted_test_loss / len(self.sigmas)
                            
                        self.tb_logger.add_scalar(f'test_sigma_{idx}_loss', test_loss.item(), global_step=step)
                        
                    self.tb_logger.add_scalar('test_total_loss', test_total_loss.item(), global_step=step)
                    self.logger.info(f"Test Total Loss: {test_total_loss.item()}")
                    
                    self.score_model.train()
                
                # Save checkpoint
                if step % self.args.snapshot_freq == 0:
                    states = [
                        self.score_model.state_dict(),
                        self.optimizer.state_dict(),
                        self.sigmas,  # Save the noise levels
                        self.lambdas   # Save the lambda weights
                    ]
                    torch.save(states, os.path.join(self.args.log_dir, f'checkpoint_{step}.pth'))
                    torch.save(states, os.path.join(self.args.log_dir, 'checkpoint.pth'))
                
                if step >= self.args.total_steps:
                    self.logger.info(f"Reached maximum steps {self.args.total_steps}. Training finished.")
                    # Save final model
                    torch.save(self.score_model.state_dict(), os.path.join(self.args.dir_name, 'model.pt'))
                    return 0
            
            # Save at the end of each epoch
            states = [
                self.score_model.state_dict(),
                self.optimizer.state_dict(),
                self.sigmas,
                self.lambdas
            ]
            torch.save(states, os.path.join(self.args.log_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Save final model
        torch.save(self.score_model.state_dict(), os.path.join(self.args.dir_name, 'model.pt'))
        self.logger.info(f"Completed {self.args.num_epochs} epochs. Training finished.")
        return 0
    
    def logit_transform(self, image, lam=1e-6):
        '''
        Apply logit transform to image data
        '''
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Training a score-based model on celebrity faces')
    
    # Architecture arguments
    parser.add_argument('--arch_name', type=str, default='ResScore', 
                        choices=['ResScore', 'UNetScore'], help='Model architecture')
    parser.add_argument('--num_channels', type=int, default=1, 
                        help='Number of channels in input images (1 for grayscale, 3 for color)')
    parser.add_argument('--RF', type=int, default=None, help='Receptive field size')
    
    # Training arguments
    parser.add_argument('--training_algo', type=str, default='dsm', choices=['ssm', 'dsm'],
                       help='Training algorithm: ssm (sliced score matching) or dsm (denoising score matching)')
    parser.add_argument('--noise_std', type=float, default=0.01, help='Noise standard deviation (for single-level training)')
    parser.add_argument('--noise_level_range', nargs=2, type=int, default=[0, 255], 
                        help='Range of noise levels (for path naming)')
    parser.add_argument('--random_noise_level', action='store_true',
                        help='Randomly select a noise level for each batch instead of using all levels')
    parser.add_argument('--noise_conditioning', action='store_true',
                        help='Use noise conditioning in the network architecture')
    parser.add_argument('--rescale_to_unit', action='store_true',
                        help='Rescale image pixel values to [0, 1] range')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--total_steps', type=int, default=1000000, help='Total training steps')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                        choices=['Adam', 'RMSProp', 'SGD'], help='Optimizer')
    parser.add_argument('--init_type', type=str, default='normal', help='Weight initialization method')
    parser.add_argument('--init_gain', type=float, default=0.02, help='Scaling factor for weight initialization')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--ngf', type=int, default=64, 
                   help='Number of filters in the UNet generator')
    parser.add_argument('--nef', type=int, default=64, 
                   help='Number of encoder filters in ResScore')
    parser.add_argument('--ndf', type=int, default=64,
                   help='Number of decoder filters in ResScore')
    parser.add_argument('--image_size', type=int, default=80,
                   help='Size of input images')
    
    # Data arguments
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--data_root_path', type=str, default='datasets/', 
                        help='Root path for datasets')
    parser.add_argument('--set_size', type=int, default=None, 
                        help='Number of images to use from dataset (defaults to half of total dataset)')
    parser.add_argument('--swap', action='store_true', 
                        help='Swap train and test sets')
    parser.add_argument('--limit_test_size', action='store_true',
                        help='Limit test set size to batch_size (for memory efficiency)')
    parser.add_argument('--logit_transform', action='store_true', 
                        help='Apply logit transform to data')
    
    # Logging and checkpoint arguments
    parser.add_argument('--dir_name', type=str, default='models/', 
                        help='Directory to save models')
    parser.add_argument('--run_dir', type=str, default='runs/', 
                        help='Directory for tensorboard runs')
    parser.add_argument('--log_dir', type=str, default='logs_swap/', 
                        help='Directory for logs')
    parser.add_argument('--experiment_name', type=str, default='celeb_face_experiment', 
                        help='Name of the experiment')
    parser.add_argument('--snapshot_freq', type=int, default=1000, 
                        help='Frequency of saving model checkpoints')
    parser.add_argument('--resume_training', action='store_true', 
                        help='Resume training from checkpoint')
    parser.add_argument('--ckpt_path', type=str, default='',
                    help='path to checkpoint_*.pth to resume from')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode with smaller dataset and fewer epochs')
    
    args = parser.parse_args()
    
    # Create runner and train
    runner = CelebFaceScoreNetRunner(args)
    runner.train()