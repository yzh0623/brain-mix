"""
Copyright (c) 2025 by paohe information technology Co., Ltd. All right reserved.
FilePath: /brain-mix/nlp/models/reasoning/fine_turning/training_monitor.py
Author: yuanzhenhui
Date: 2025-09-16 11:33:49
LastEditTime: 2025-09-16 11:35:12
"""

class TrainingMonitor:
    
    def __init__(self):
        self.metrics = []
        
    def log_metrics(self, epoch: int, loss: float, learning_rate: float, gpu_memory: float):
        """
        Log the metrics for the current epoch.

        Args:
            epoch (int): The current epoch.
            loss (float): The current loss.
            learning_rate (float): The current learning rate.
            gpu_memory (float): The current GPU memory usage in GB.
        """
        self.metrics.append({
            'epoch': epoch,
            'loss': loss,
            'learning_rate': learning_rate,
            'gpu_memory_gb': gpu_memory
        })
        
    def get_summary(self):
        """
        Get a summary of the training process.

        Returns a dictionary containing the best loss, final loss, and convergence rate.

        :return: A dictionary containing the summary of the training process.
        :rtype: dict
        """
        if not self.metrics:
            return {}
        
        # Extract the losses from the metrics
        losses = [m['loss'] for m in self.metrics]
        
        # Calculate the best loss, final loss, and convergence rate
        best_loss = min(losses)
        final_loss = losses[-1]
        convergence_rate = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
        
        # Return the summary as a dictionary
        return {
            'best_loss': best_loss,
            'final_loss': final_loss,
            'convergence_rate': convergence_rate
        }
