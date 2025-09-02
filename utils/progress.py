import logging
from typing import Optional, Callable
from rich.progress import Progress, TaskID, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from config.cache import get_cache_settings

logger = logging.getLogger(__name__)

class RaptorProgress:
    """
    Non-blocking progress bar for RAPTOR tree building
    Optimized to not impact API performance
    """
    
    def __init__(self, description: str = "RAPTOR Processing", show_console: bool = True):
        self.show_console = show_console
        self.progress: Optional[Progress] = None
        self.task_id: Optional[TaskID] = None
        self.description = description
        
        # Only show progress if cache logging is enabled (performance control)
        settings = get_cache_settings()
        self.enabled = settings.cache_hit_log_enabled or show_console
        
    def __enter__(self):
        if not self.enabled:
            return self
            
        try:
            self.progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                TimeElapsedColumn(),
                "•", 
                TimeRemainingColumn(),
                console=Console(stderr=True),  # Use stderr to not interfere with API responses
                disable=not self.show_console
            )
            self.progress.start()
            self.task_id = self.progress.add_task(self.description, total=100)
            
        except Exception as e:
            logger.warning(f"Progress bar initialization failed: {e}")
            self.enabled = False
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            try:
                self.progress.stop()
            except Exception as e:
                logger.warning(f"Progress bar cleanup failed: {e}")
                
    def update(self, advance: float = 0, completed: Optional[float] = None, description: Optional[str] = None):
        """Update progress (non-blocking)"""
        if not self.enabled or not self.progress or self.task_id is None:
            return
            
        try:
            update_kwargs = {}
            if advance > 0:
                update_kwargs['advance'] = advance
            if completed is not None:
                update_kwargs['completed'] = completed
            if description:
                update_kwargs['description'] = description
                
            self.progress.update(self.task_id, **update_kwargs)
            
        except Exception as e:
            logger.warning(f"Progress update failed: {e}")
            
    def set_total(self, total: int):
        """Set total steps for progress calculation"""
        if not self.enabled or not self.progress or self.task_id is None:
            return
            
        try:
            self.progress.update(self.task_id, total=total)
        except Exception as e:
            logger.warning(f"Progress total update failed: {e}")


def create_raptor_progress_callback(progress_bar: Optional[RaptorProgress] = None) -> Callable[[str], None]:
    """
    Create a callback function for RAPTOR tree building progress
    Returns a non-blocking callback that updates progress bar
    """
    current_step = 0
    
    def callback(msg: str) -> None:
        nonlocal current_step
        current_step += 1
        
        if progress_bar:
            # Extract numbers from message for better progress calculation
            if "→" in msg:
                try:
                    # Parse "Layer X: Y → Z clusters" format
                    parts = msg.split(":")
                    if len(parts) > 1:
                        layer_info = parts[1].strip()
                        if "→" in layer_info:
                            before, after = layer_info.split("→")
                            before_count = int(before.strip().split()[0])
                            after_count = int(after.strip().split()[0])
                            
                            # Calculate progress based on reduction ratio
                            reduction_ratio = after_count / before_count if before_count > 0 else 0
                            progress_increment = 100 / max(10, before_count) * (1 - reduction_ratio)
                            
                            progress_bar.update(advance=progress_increment, description=f"🌳 {msg}")
                            return
                except (ValueError, IndexError):
                    pass
            
            # Fallback: simple step-based progress
            progress_bar.update(advance=10, description=f"🌳 {msg}")
        
        # Always log progress (for server logs)
        logger.info(f"📊 RAPTOR Progress: {msg}")
    
    return callback


def create_chunking_progress_callback(progress_bar: Optional[RaptorProgress] = None) -> Callable[[str], None]:
    """
    Create a callback function for document chunking progress
    Returns a non-blocking callback that updates progress bar
    """
    current_step = 0
    total_steps = 6  # Chunking has fewer steps than RAPTOR
    
    def callback(msg: str) -> None:
        nonlocal current_step
        current_step += 1
        
        if progress_bar:
            # Calculate progress percentage
            progress_percent = (current_step / total_steps) * 100
            
            # Extract chunk count if available
            if "chunks" in msg.lower():
                try:
                    # Parse "Generated X chunks" or "Processing X sections" 
                    import re
                    numbers = re.findall(r'\d+', msg)
                    if numbers:
                        chunk_count = int(numbers[0])
                        # More chunks = more progress
                        progress_increment = min(20, chunk_count / 10)
                        progress_bar.update(advance=progress_increment, description=f"📄 {msg}")
                        return
                except (ValueError, IndexError):
                    pass
            
            # Fallback: step-based progress for chunking
            progress_increment = 100 / total_steps
            progress_bar.update(advance=progress_increment, description=f"📄 {msg}")
        
        # Always log progress (for server logs)
        logger.info(f"📄 Chunking Progress: {msg}")
    
    return callback


# Convenience functions
def raptor_progress_context(description: str = "Building RAPTOR Tree", show_console: bool = False):
    """Context manager for RAPTOR progress tracking"""
    return RaptorProgress(description=description, show_console=show_console)

def chunking_progress_context(description: str = "Processing Document", show_console: bool = False):
    """Context manager for document chunking progress tracking"""
    return RaptorProgress(description=description, show_console=show_console)

