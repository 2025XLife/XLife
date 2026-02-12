"""
HF-mirror configuration management module
Used to manage mirror download settings for Hugging Face models
"""

import os
from typing import Optional

class HFMirrorConfig:
    """HF-mirror configuration management class"""
    
    def __init__(self, 
                 mirror_url: str = "https://hf-mirror.com",
                 enabled: bool = True):
        """
        Initialize HF-mirror configuration
        
        Args:
            mirror_url: Mirror site URL
            enabled: Whether to enable mirror
        """
        self.mirror_url = mirror_url
        self.enabled = enabled
        self.original_endpoint = os.environ.get('HF_ENDPOINT', None)
        
    def enable_mirror(self):
        """Enable HF-mirror"""
        if self.enabled:
            os.environ['HF_ENDPOINT'] = self.mirror_url
            print(f"Enabled HF-mirror: {self.mirror_url}")
        else:
            print("HF-mirror is disabled")
            
    def disable_mirror(self):
        """Disable HF-mirror, restore original settings"""
        if self.original_endpoint is not None:
            os.environ['HF_ENDPOINT'] = self.original_endpoint
        elif 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        print("Disabled HF-mirror, restored original settings")
        
    def get_status(self) -> dict:
        """Get current configuration status"""
        return {
            "mirror_url": self.mirror_url,
            "enabled": self.enabled,
            "current_endpoint": os.environ.get('HF_ENDPOINT', 'Not set'),
            "is_mirror_active": os.environ.get('HF_ENDPOINT') == self.mirror_url
        }
        
    def __enter__(self):
        """Context manager entry"""
        self.enable_mirror()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disable_mirror()


# Global configuration instance
_global_config = None

def get_hf_mirror_config(mirror_url: str = "https://hf-mirror.com", 
                        enabled: bool = True) -> HFMirrorConfig:
    """
    Get global HF-mirror configuration
    
    Args:
        mirror_url: Mirror site URL
        enabled: Whether to enable mirror
        
    Returns:
        HFMirrorConfig: Configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = HFMirrorConfig(mirror_url, enabled)
    return _global_config


def enable_hf_mirror(mirror_url: str = "https://hf-mirror.com"):
    """
    Enable HF-mirror
    
    Args:
        mirror_url: Mirror site URL
    """
    config = get_hf_mirror_config(mirror_url, True)
    config.enable_mirror()
    return config


def disable_hf_mirror():
    """Disable HF-mirror"""
    global _global_config
    if _global_config is not None:
        _global_config.disable_mirror()


def get_hf_mirror_status() -> dict:
    """Get HF-mirror status"""
    global _global_config
    if _global_config is None:
        return {"status": "Not initialized"}
    return _global_config.get_status()


def with_hf_mirror(mirror_url: str = "https://hf-mirror.com"):
    """
    Create HF-mirror context manager
    
    Args:
        mirror_url: Mirror site URL
        
    Returns:
        HFMirrorConfig: Configuration instance
    """
    return HFMirrorConfig(mirror_url, True)


if __name__ == "__main__":
    print("HF-mirror configuration management test")
    print("=" * 50)
    
    config = get_hf_mirror_config()
    print("Initial status:", config.get_status())
    
    config.enable_mirror()
    print("After enabling:", config.get_status())
    
    config.disable_mirror()
    print("After disabling:", config.get_status())
    
    print("\nTesting context manager:")
    with with_hf_mirror() as cfg:
        print("Inside context:", cfg.get_status())
    print("Outside context:", cfg.get_status())