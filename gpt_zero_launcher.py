#!/usr/bin/env python3
"""
GPT-Ø Somnus Sovereign Kernel Launcher
======================================

Production launcher for GPT-Ø with full Somnus Sovereign Kernel integration.
Orchestrates all components through the CAS system for seamless operation.

Features:
- Complete Somnus CAS integration
- 5-tier neural memory runtime
- Constitutional AI framework
- Model assimilation capabilities
- Performance monitoring and optimization
- Graceful error handling and recovery

Author: Morpheus
Version: 1.0.0-alpha
"""

import asyncio
import logging
import signal
import sys
import traceback
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.status import Status
from rich.logging import RichHandler

# Somnus integration
from gpt_zero_integration import (
    GPTZeroCAIntegrationBridge, 
    GPTZeroIntegrationConfig,
    create_gpt_zero_integration
)

# Configure rich console
console = Console()

# Configure logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

@dataclass
class LauncherConfig:
    """Configuration for the Somnus GPT-Ø launcher"""
    cas_config_path: str = "gpt_zero.cas.yml"
    neural_config_path: str = "gpt_zero_neural_config.yaml"
    agent_config_path: str = "agent_config.yaml"
    max_memory_gb: float = 8.0
    enable_debug_mode: bool = False
    auto_start_performance_monitoring: bool = True
    graceful_shutdown_timeout: int = 30

class SomnusGPTZeroLauncher:
    """
    Main launcher class for GPT-Ø with Somnus integration.
    Handles initialization, monitoring, and shutdown.
    """
    
    def __init__(self, config: LauncherConfig):
        self.config = config
        self.integration_bridge: Optional[GPTZeroCAIntegrationBridge] = None
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance metrics
        self.start_time: Optional[datetime] = None
        self.metrics: Dict[str, Any] = {}
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self._graceful_shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> bool:
        """
        Initialize GPT-Ø with full Somnus integration.
        
        Returns:
            bool: True if initialization successful
        """
        
        self.start_time = datetime.now()
        
        # Display startup banner
        self._display_startup_banner()
        
        with Status("[bold blue]Initializing GPT-Ø Somnus Integration...", console=console):
            try:
                # Validate configuration files
                await self._validate_configuration_files()
                
                # Create integration bridge
                integration_config = GPTZeroIntegrationConfig(
                    cas_config_path=self.config.cas_config_path,
                    neural_config_path=self.config.neural_config_path,
                    max_memory_gb=self.config.max_memory_gb,
                    enable_model_assimilation=True,
                    enable_self_modification=True
                )
                
                # Initialize with progress tracking
                self.integration_bridge = await self._initialize_with_progress(integration_config)
                
                if not self.integration_bridge:
                    raise RuntimeError("Failed to initialize integration bridge")
                
                # Verify successful initialization
                status = await self.integration_bridge.get_integration_status()
                if not status['initialized']:
                    raise RuntimeError("Integration bridge not properly initialized")
                
                self.running = True
                
                # Display success information
                self._display_success_information(status)
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Initialization failed: {e}")
                console.print_exception()
                return False
    
    async def _validate_configuration_files(self):
        """Validate all required configuration files exist"""
        
        required_files = [
            self.config.cas_config_path,
            self.config.neural_config_path,
            self.config.agent_config_path
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise FileNotFoundError(f"Missing configuration files: {missing_files}")
        
        logger.info("✅ All configuration files validated")
    
    async def _initialize_with_progress(self, integration_config: GPTZeroIntegrationConfig) -> Optional[GPTZeroCAIntegrationBridge]:
        """Initialize with detailed progress tracking"""
        
        progress_steps = [
            "Validating Somnus Infrastructure",
            "Loading CAS Specification", 
            "Initializing Neural Memory Runtime",
            "Starting GPT-Ø Core Components",
            "Setting up Constitutional Framework",
            "Initializing Model Assimilation",
            "Registering with Neural Manager",
            "Starting Performance Monitoring"
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Initializing...", total=len(progress_steps))
            
            for i, step in enumerate(progress_steps):
                progress.update(task, description=f"[bold blue]{step}...")
                await asyncio.sleep(0.5)  # Visual delay for progress display
                progress.advance(task)
            
            # Actually create the integration
            bridge = await create_gpt_zero_integration(
                cas_config_path=integration_config.cas_config_path,
                neural_config_path=integration_config.neural_config_path,
                max_memory_gb=integration_config.max_memory_gb,
                enable_model_assimilation=integration_config.enable_model_assimilation,
                enable_self_modification=integration_config.enable_self_modification
            )
            
            progress.update(task, description="[bold green]✅ Initialization Complete!")
            
        return bridge
    
    def _display_startup_banner(self):
        """Display the startup banner"""
        
        banner_text = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    GPT-Ø: Revolutionary Self-Modifying Multimodal AI          ║
    ║     Somnus Sovereign Kernel Integration                       ║
    ║                                                               ║
    ║     33B+ Parameters •  Self-Modifying •  16+ Modalities       ║
    ║     8GB RAM •  Constitutional AI •  Model Assimilation        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
        """
        
        console.print(Panel(
            banner_text,
            style="bold blue",
            border_style="bright_blue"
        ))
    
    def _display_success_information(self, status: Dict[str, Any]):
        """Display successful initialization information"""
        
        # Create information table
        table = Table(title="🎉 GPT-Ø Successfully Initialized", style="green")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")
        
        # Add component status
        components = status.get('components', {})
        
        table.add_row(
            "🧠 GPT-Ø Model",
            "✅ Active" if components.get('gpt_model') else "❌ Failed",
            "33B+ parameter self-modifying transformer"
        )
        
        table.add_row(
            "🧬 Neural Memory",
            "✅ Active" if components.get('neural_memory') else "❌ Failed", 
            "5-tier hierarchical memory system"
        )
        
        table.add_row(
            "⚖️ Constitutional AI",
            "✅ Active" if components.get('constitutional_governor') else "❌ Failed",
            "Safety and compliance framework"
        )
        
        table.add_row(
            "🔄 GGUF Assimilator",
            "✅ Active" if components.get('gguf_assimilator') else "❌ Failed",
            "Universal model assimilation system"
        )
        
        # Output heads
        output_heads = components.get('output_heads', {})
        tool_status = "✅ Active" if output_heads.get('tool_head') else "❌ Failed"
        isr_status = "✅ Active" if output_heads.get('isr_head') else "❌ Failed"
        spatial_status = "✅ Active" if output_heads.get('ears_head') else "❌ Failed"
        
        table.add_row("🔧 Tool Head", tool_status, "Universal tool synthesis")
        table.add_row("👁️ ISR Head", isr_status, "Intelligence & surveillance")
        table.add_row("🌍 Spatial Head", spatial_status, "Spatial domain processing")
        
        console.print(table)
        
        # Display memory information
        memory_info = status.get('performance_metrics', {}).get('memory_usage', {})
        if memory_info:
            console.print(f"\n💾 Memory Usage: {memory_info}")
        
        # Display startup time
        if self.start_time:
            startup_time = (datetime.now() - self.start_time).total_seconds()
            console.print(f"⚡ Startup Time: {startup_time:.2f} seconds")
    
    async def run(self):
        """Main run loop for GPT-Ø"""
        
        if not self.running or not self.integration_bridge:
            logger.error("Cannot run: GPT-Ø not properly initialized")
            return
        
        logger.info("🚀 GPT-Ø is now running with Somnus integration!")
        logger.info("Press Ctrl+C to gracefully shutdown")
        
        try:
            # Start performance monitoring if enabled
            if self.config.auto_start_performance_monitoring:
                asyncio.create_task(self._performance_monitoring_loop())
            
            # Main service loop
            while self.running and not self.shutdown_event.is_set():
                # Monitor integration status
                status = await self.integration_bridge.get_integration_status()
                
                if not status['initialized']:
                    logger.error("Integration bridge lost initialization state")
                    break
                
                # Update metrics
                self.metrics.update(status.get('performance_metrics', {}))
                
                # Sleep and wait for shutdown signal
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=30.0)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Continue monitoring
                
        except Exception as e:
            logger.error(f"Runtime error: {e}")
            console.print_exception()
        finally:
            await self._graceful_shutdown()
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring loop"""
        
        logger.info("📊 Starting performance monitoring...")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                if self.integration_bridge:
                    status = await self.integration_bridge.get_integration_status()
                    metrics = status.get('performance_metrics', {})
                    
                    # Log key metrics periodically
                    if metrics:
                        memory_usage = metrics.get('memory_usage', {})
                        if memory_usage:
                            logger.debug(f"Memory tiers: {memory_usage}")
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _graceful_shutdown(self):
        """Perform graceful shutdown"""
        
        if not self.running:
            return
        
        self.running = False
        self.shutdown_event.set()
        
        console.print("\n🛑 [bold yellow]Initiating graceful shutdown...")
        
        with Status("[bold yellow]Shutting down GPT-Ø...", console=console):
            try:
                if self.integration_bridge:
                    await asyncio.wait_for(
                        self.integration_bridge.shutdown(),
                        timeout=self.config.graceful_shutdown_timeout
                    )
                
                # Save final metrics
                await self._save_final_metrics()
                
            except asyncio.TimeoutError:
                logger.warning("Shutdown timeout exceeded, forcing exit")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
            finally:
                console.print("✅ [bold green]GPT-Ø shutdown complete")
    
    async def _save_final_metrics(self):
        """Save final performance metrics"""
        
        if not self.metrics:
            return
        
        try:
            metrics_path = Path("./metrics/final_session_metrics.json")
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            
            final_metrics = {
                'session_start': self.start_time.isoformat() if self.start_time else None,
                'session_end': datetime.now().isoformat(),
                'runtime_metrics': self.metrics,
                'launcher_config': {
                    'cas_config_path': self.config.cas_config_path,
                    'neural_config_path': self.config.neural_config_path,
                    'max_memory_gb': self.config.max_memory_gb
                }
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=2, default=str)
            
            logger.info(f"📊 Final metrics saved to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Failed to save final metrics: {e}")

async def main():
    """Main entry point"""
    
    # Parse command line arguments (basic implementation)
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-Ø Somnus Launcher")
    parser.add_argument("--cas-config", default="gpt_zero.cas.yml", help="CAS configuration file")
    parser.add_argument("--neural-config", default="gpt_zero_neural_config.yaml", help="Neural configuration file")
    parser.add_argument("--agent-config", default="agent_config.yaml", help="Agent configuration file")
    parser.add_argument("--max-memory", type=float, default=8.0, help="Maximum memory in GB")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create launcher configuration
    launcher_config = LauncherConfig(
        cas_config_path=args.cas_config,
        neural_config_path=args.neural_config,
        agent_config_path=args.agent_config,
        max_memory_gb=args.max_memory,
        enable_debug_mode=args.debug
    )
    
    # Create and run launcher
    launcher = SomnusGPTZeroLauncher(launcher_config)
    
    try:
        # Initialize
        success = await launcher.initialize()
        if not success:
            console.print("❌ [bold red]Failed to initialize GPT-Ø")
            sys.exit(1)
        
        # Run main loop
        await launcher.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        console.print_exception()
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())