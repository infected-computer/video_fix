"""
Enhanced CLI Interface for PhoenixDRS
Advanced command-line interface with rich formatting, interactive modes, and comprehensive features
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

# Rich library for enhanced terminal output
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live
from rich.spinner import Spinner
from rich import box

# Command parsing and completion
import click
from click_completion import init as click_completion_init
import cmd2
from cmd2 import with_argparser, with_category

# Configuration and data handling
import yaml
import toml
import configparser

# HTTP client for API calls
import httpx
import aiohttp

# File handling and utilities
import magic
import mimetypes
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)
console = Console()


class CLIConfig:
    """Configuration manager for CLI"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self.get_default_config_path()
        self.config = self.load_config()
    
    def get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        config_dir = Path.home() / ".phoenixdrs"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "cli_config.yaml"
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_path.exists():
            self.create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            return self.get_default_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "api": {
                "base_url": "http://localhost:8000",
                "api_key": "",
                "timeout": 30,
                "retry_count": 3
            },
            "cli": {
                "output_format": "table",
                "color_theme": "auto",
                "pager": "auto",
                "history_size": 1000,
                "auto_complete": True
            },
            "logging": {
                "level": "INFO",
                "file": "~/.phoenixdrs/cli.log",
                "max_size": "10MB",
                "backup_count": 5
            },
            "defaults": {
                "analysis_types": ["all"],
                "output_directory": "~/phoenixdrs_output",
                "temp_directory": "/tmp/phoenixdrs",
                "parallel_jobs": 4
            }
        }
    
    def create_default_config(self):
        """Create default configuration file"""
        self.config = self.get_default_config()
        self.save_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()


class APIClient:
    """HTTP client for API communication"""
    
    def __init__(self, config: CLIConfig):
        self.config = config
        self.base_url = config.get("api.base_url")
        self.api_key = config.get("api.api_key")
        self.timeout = config.get("api.timeout", 30)
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make GET request"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise Exception(f"API request failed: {e}")
    
    async def post(self, endpoint: str, data: Dict[str, Any] = None, 
                   json_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make POST request"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.post(url, data=data, json=json_data) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise Exception(f"API request failed: {e}")
    
    async def upload_file(self, endpoint: str, file_path: str, 
                         additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload file to API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            with open(file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=Path(file_path).name)
                
                if additional_data:
                    for key, value in additional_data.items():
                        data.add_field(key, str(value))
                
                async with self.session.post(url, data=data) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            raise Exception(f"File upload failed: {e}")


class OutputFormatter:
    """Format output in various styles"""
    
    def __init__(self, config: CLIConfig):
        self.config = config
        self.format_type = config.get("cli.output_format", "table")
    
    def format_data(self, data: Any, format_type: Optional[str] = None) -> str:
        """Format data based on configured format"""
        fmt = format_type or self.format_type
        
        if fmt == "json":
            return self.format_json(data)
        elif fmt == "yaml":
            return self.format_yaml(data)
        elif fmt == "table":
            return self.format_table(data)
        elif fmt == "tree":
            return self.format_tree(data)
        else:
            return str(data)
    
    def format_json(self, data: Any) -> str:
        """Format as JSON"""
        return json.dumps(data, indent=2, default=str)
    
    def format_yaml(self, data: Any) -> str:
        """Format as YAML"""
        return yaml.dump(data, default_flow_style=False)
    
    def format_table(self, data: Any) -> Table:
        """Format as Rich table"""
        if isinstance(data, list) and data:
            table = Table(show_header=True, header_style="bold magenta")
            
            # Use first item to determine columns
            if isinstance(data[0], dict):
                for key in data[0].keys():
                    table.add_column(str(key).title())
                
                for item in data:
                    row = [str(item.get(key, "")) for key in data[0].keys()]
                    table.add_row(*row)
            else:
                table.add_column("Value")
                for item in data:
                    table.add_row(str(item))
            
            return table
        
        elif isinstance(data, dict):
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Key")
            table.add_column("Value")
            
            for key, value in data.items():
                table.add_row(str(key), str(value))
            
            return table
        
        else:
            return Table.from_dict({"Data": [str(data)]})
    
    def format_tree(self, data: Any, name: str = "Data") -> Tree:
        """Format as Rich tree"""
        tree = Tree(name)
        self._add_tree_items(tree, data)
        return tree
    
    def _add_tree_items(self, parent: Tree, data: Any):
        """Recursively add items to tree"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    branch = parent.add(str(key))
                    self._add_tree_items(branch, value)
                else:
                    parent.add(f"{key}: {value}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    branch = parent.add(f"[{i}]")
                    self._add_tree_items(branch, item)
                else:
                    parent.add(f"[{i}]: {item}")
        
        else:
            parent.add(str(data))


class InteractiveShell(cmd2.Cmd):
    """Interactive shell for PhoenixDRS CLI"""
    
    def __init__(self, config: CLIConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.config = config
        self.formatter = OutputFormatter(config)
        self.api_client = None
        
        # Shell configuration
        self.intro = self.get_intro_text()
        self.prompt = "(phoenixdrs) "
        self.allow_cli_args = False
        
        # Command categories
        self.add_settable(cmd2.Settable('output_format', str, 'Output format (table, json, yaml, tree)', 
                                       choices=['table', 'json', 'yaml', 'tree']))
        
        # Set up completion
        self.complete_analyze = self._complete_file_path
        self.complete_upload = self._complete_file_path
    
    def get_intro_text(self) -> str:
        """Get shell introduction text"""
        intro = Panel.fit(
            "[bold blue]PhoenixDRS Professional 2.0[/bold blue]\n"
            "[italic]Digital Forensics and Data Recovery System[/italic]\n\n"
            "Type 'help' for available commands.\n"
            "Type 'exit' or 'quit' to exit the shell.",
            border_style="blue"
        )
        
        with console.capture() as capture:
            console.print(intro)
        
        return capture.get()
    
    # Configuration commands
    config_parser = argparse.ArgumentParser()
    config_subparsers = config_parser.add_subparsers(dest='action', help='Configuration actions')
    
    config_get_parser = config_subparsers.add_parser('get', help='Get configuration value')
    config_get_parser.add_argument('key', help='Configuration key')
    
    config_set_parser = config_subparsers.add_parser('set', help='Set configuration value')
    config_set_parser.add_argument('key', help='Configuration key')
    config_set_parser.add_argument('value', help='Configuration value')
    
    config_list_parser = config_subparsers.add_parser('list', help='List all configuration')
    
    @with_argparser(config_parser)
    @with_category("Configuration")
    def do_config(self, args):
        """Manage CLI configuration"""
        if args.action == 'get':
            value = self.config.get(args.key)
            if value is not None:
                console.print(f"[bold]{args.key}[/bold]: {value}")
            else:
                console.print(f"[red]Configuration key '{args.key}' not found[/red]")
        
        elif args.action == 'set':
            # Try to parse value as JSON, fall back to string
            try:
                value = json.loads(args.value)
            except json.JSONDecodeError:
                value = args.value
            
            self.config.set(args.key, value)
            console.print(f"[green]Set {args.key} = {value}[/green]")
        
        elif args.action == 'list':
            formatted = self.formatter.format_data(self.config.config)
            if isinstance(formatted, (Table, Tree)):
                console.print(formatted)
            else:
                console.print(formatted)
    
    # Analysis commands
    analyze_parser = argparse.ArgumentParser()
    analyze_parser.add_argument('file_path', help='Path to file to analyze')
    analyze_parser.add_argument('--types', nargs='+', default=['all'], 
                               help='Analysis types to perform')
    analyze_parser.add_argument('--deep', action='store_true', help='Perform deep analysis')
    analyze_parser.add_argument('--output', '-o', help='Output file for results')
    analyze_parser.add_argument('--format', choices=['json', 'yaml', 'table'], 
                               help='Output format')
    
    @with_argparser(analyze_parser)
    @with_category("Analysis")
    def do_analyze(self, args):
        """Analyze a file"""
        if not Path(args.file_path).exists():
            console.print(f"[red]File not found: {args.file_path}[/red]")
            return
        
        asyncio.run(self._analyze_file_async(args))
    
    async def _analyze_file_async(self, args):
        """Async file analysis"""
        async with APIClient(self.config) as client:
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Analyzing file...", total=None)
                    
                    # Submit analysis request
                    request_data = {
                        "file_path": args.file_path,
                        "analysis_types": args.types,
                        "deep_scan": args.deep,
                        "include_metadata": True
                    }
                    
                    response = await client.post("/api/v1/analyze/file", json_data=request_data)
                    operation_id = response.get("operation_id")
                    
                    if not operation_id:
                        console.print("[red]Failed to start analysis[/red]")
                        return
                    
                    # Poll for results
                    while True:
                        result_response = await client.get(f"/api/v1/analyze/{operation_id}")
                        status = result_response.get("status")
                        
                        if status == "completed":
                            progress.update(task, description="Analysis completed")
                            break
                        elif status == "failed":
                            console.print("[red]Analysis failed[/red]")
                            return
                        
                        await asyncio.sleep(1)
                
                # Display results
                self._display_analysis_results(result_response, args.format)
                
                # Save to file if requested
                if args.output:
                    self._save_results(result_response, args.output, args.format)
                
            except Exception as e:
                console.print(f"[red]Analysis error: {e}[/red]")
    
    def _display_analysis_results(self, results: Dict[str, Any], format_type: Optional[str] = None):
        """Display analysis results"""
        console.print("\n[bold green]Analysis Results:[/bold green]")
        
        # Basic info
        info_table = Table(title="File Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("File Path", results.get("file_path", ""))
        info_table.add_row("Status", results.get("status", ""))
        info_table.add_row("Operation ID", results.get("operation_id", ""))
        info_table.add_row("Execution Time", f"{results.get('execution_time', 0):.2f}s")
        
        console.print(info_table)
        
        # Analysis results
        analysis_results = results.get("results", {})
        if analysis_results:
            console.print("\n[bold blue]Analysis Details:[/bold blue]")
            formatted = self.formatter.format_data(analysis_results, format_type)
            
            if isinstance(formatted, (Table, Tree)):
                console.print(formatted)
            else:
                syntax = Syntax(formatted, "json" if format_type == "json" else "yaml", 
                              theme="monokai", line_numbers=True)
                console.print(syntax)
        
        # Confidence scores
        confidence_scores = results.get("confidence_scores", {})
        if confidence_scores:
            console.print("\n[bold yellow]Confidence Scores:[/bold yellow]")
            conf_table = Table()
            conf_table.add_column("Analysis Type", style="cyan")
            conf_table.add_column("Confidence", style="green")
            
            for analysis_type, score in confidence_scores.items():
                conf_table.add_row(analysis_type, f"{score:.2%}")
            
            console.print(conf_table)
    
    def _save_results(self, results: Dict[str, Any], output_path: str, format_type: Optional[str] = None):
        """Save results to file"""
        try:
            output_file = Path(output_path)
            
            if format_type == "json" or output_file.suffix.lower() == ".json":
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format_type == "yaml" or output_file.suffix.lower() in [".yaml", ".yml"]:
                with open(output_file, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False)
            else:
                with open(output_file, 'w') as f:
                    f.write(str(results))
            
            console.print(f"[green]Results saved to: {output_file}[/green]")
        
        except Exception as e:
            console.print(f"[red]Failed to save results: {e}[/red]")
    
    # Upload commands
    upload_parser = argparse.ArgumentParser()
    upload_parser.add_argument('file_path', help='Path to file to upload')
    upload_parser.add_argument('--description', help='File description')
    upload_parser.add_argument('--tags', nargs='+', help='File tags')
    
    @with_argparser(upload_parser)
    @with_category("File Management")
    def do_upload(self, args):
        """Upload a file for analysis"""
        if not Path(args.file_path).exists():
            console.print(f"[red]File not found: {args.file_path}[/red]")
            return
        
        asyncio.run(self._upload_file_async(args))
    
    async def _upload_file_async(self, args):
        """Async file upload"""
        async with APIClient(self.config) as client:
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Uploading file...", total=100)
                    
                    additional_data = {}
                    if args.description:
                        additional_data['description'] = args.description
                    if args.tags:
                        additional_data['tags'] = ','.join(args.tags)
                    
                    response = await client.upload_file(
                        "/api/v1/upload", 
                        args.file_path, 
                        additional_data
                    )
                    
                    progress.update(task, completed=100)
                
                console.print("[green]File uploaded successfully![/green]")
                
                # Display upload info
                upload_table = Table(title="Upload Information")
                upload_table.add_column("Property", style="cyan")
                upload_table.add_column("Value", style="white")
                
                for key, value in response.items():
                    upload_table.add_row(str(key).title(), str(value))
                
                console.print(upload_table)
                
            except Exception as e:
                console.print(f"[red]Upload error: {e}[/red]")
    
    # Workflow commands
    workflow_parser = argparse.ArgumentParser()
    workflow_subparsers = workflow_parser.add_subparsers(dest='action', help='Workflow actions')
    
    workflow_list_parser = workflow_subparsers.add_parser('list', help='List workflows')
    workflow_create_parser = workflow_subparsers.add_parser('create', help='Create workflow')
    workflow_create_parser.add_argument('name', help='Workflow name')
    workflow_create_parser.add_argument('--type', required=True, help='Workflow type')
    workflow_create_parser.add_argument('--config', help='Configuration file')
    
    workflow_status_parser = workflow_subparsers.add_parser('status', help='Get workflow status')
    workflow_status_parser.add_argument('workflow_id', help='Workflow ID')
    
    @with_argparser(workflow_parser)
    @with_category("Workflow Management")
    def do_workflow(self, args):
        """Manage workflows"""
        asyncio.run(self._workflow_command_async(args))
    
    async def _workflow_command_async(self, args):
        """Async workflow command handler"""
        async with APIClient(self.config) as client:
            try:
                if args.action == 'list':
                    # List workflows (placeholder implementation)
                    console.print("[yellow]Workflow listing not yet implemented[/yellow]")
                
                elif args.action == 'create':
                    # Create workflow
                    workflow_data = {
                        "name": args.name,
                        "workflow_type": args.type,
                        "tasks": [],  # Would be loaded from config file
                        "configuration": {}
                    }
                    
                    if args.config and Path(args.config).exists():
                        with open(args.config, 'r') as f:
                            config_data = yaml.safe_load(f)
                            workflow_data.update(config_data)
                    
                    response = await client.post("/api/v1/workflow", json_data=workflow_data)
                    console.print("[green]Workflow created successfully![/green]")
                    console.print(f"Workflow ID: {response.get('workflow_id')}")
                
                elif args.action == 'status':
                    # Get workflow status
                    response = await client.get(f"/api/v1/workflow/{args.workflow_id}")
                    
                    status_table = Table(title=f"Workflow Status - {args.workflow_id}")
                    status_table.add_column("Property", style="cyan")
                    status_table.add_column("Value", style="white")
                    
                    for key, value in response.items():
                        status_table.add_row(str(key).title(), str(value))
                    
                    console.print(status_table)
                
            except Exception as e:
                console.print(f"[red]Workflow command error: {e}[/red]")
    
    # System commands
    @with_category("System")
    def do_status(self, _):
        """Get system status"""
        asyncio.run(self._get_system_status_async())
    
    async def _get_system_status_async(self):
        """Async system status"""
        async with APIClient(self.config) as client:
            try:
                response = await client.get("/api/v1/status")
                
                status_panel = Panel(
                    self.formatter.format_table(response),
                    title="System Status",
                    border_style="green"
                )
                
                console.print(status_panel)
                
            except Exception as e:
                console.print(f"[red]Status error: {e}[/red]")
    
    @with_category("System")
    def do_plugins(self, _):
        """List available plugins"""
        asyncio.run(self._list_plugins_async())
    
    async def _list_plugins_async(self):
        """Async plugin listing"""
        async with APIClient(self.config) as client:
            try:
                response = await client.get("/api/v1/plugins")
                
                if not response:
                    console.print("[yellow]No plugins found[/yellow]")
                    return
                
                plugins_table = Table(title="Available Plugins")
                plugins_table.add_column("Name", style="cyan")
                plugins_table.add_column("Version", style="white")
                plugins_table.add_column("Type", style="yellow")
                plugins_table.add_column("Status", style="green")
                
                for plugin in response:
                    status = "Active" if plugin.get("is_active") else "Inactive"
                    plugins_table.add_row(
                        plugin.get("name", ""),
                        plugin.get("version", ""),
                        plugin.get("plugin_type", ""),
                        status
                    )
                
                console.print(plugins_table)
                
            except Exception as e:
                console.print(f"[red]Plugins error: {e}[/red]")
    
    # Utility methods
    def _complete_file_path(self, text, line, begidx, endidx):
        """File path completion"""
        return cmd2.utils.basic_complete(text, line, begidx, endidx, 
                                        match_against=self._get_file_completions(text))
    
    def _get_file_completions(self, text: str) -> List[str]:
        """Get file completions for path"""
        try:
            path = Path(text)
            if path.is_dir():
                return [str(p) for p in path.iterdir()]
            else:
                parent = path.parent
                if parent.exists():
                    return [str(p) for p in parent.iterdir() 
                           if str(p).startswith(text)]
        except Exception:
            pass
        
        return []
    
    def do_clear(self, _):
        """Clear the terminal screen"""
        console.clear()
    
    def do_version(self, _):
        """Show version information"""
        version_info = Panel(
            "[bold blue]PhoenixDRS Professional 2.0.0[/bold blue]\n"
            "[italic]Digital Forensics and Data Recovery System[/italic]\n\n"
            "CLI Version: 2.0.0\n"
            "API Version: 2.0.0\n"
            "Build Date: 2024-07-24",
            title="Version Information",
            border_style="blue"
        )
        console.print(version_info)


def create_click_cli(config: CLIConfig) -> click.Group:
    """Create Click-based CLI interface"""
    
    @click.group()
    @click.option('--config', '-c', help='Configuration file path')
    @click.option('--verbose', '-v', is_flag=True, help='Verbose output')
    @click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml', 'tree']), 
                  help='Output format')
    @click.pass_context
    def cli(ctx, config, verbose, format):
        """PhoenixDRS Professional CLI"""
        ctx.ensure_object(dict)
        
        # Initialize configuration
        if config:
            ctx.obj['config'] = CLIConfig(config)
        else:
            ctx.obj['config'] = CLIConfig()
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        # Set output format
        if format:
            ctx.obj['config'].set('cli.output_format', format)
        
        ctx.obj['formatter'] = OutputFormatter(ctx.obj['config'])
    
    @cli.command()
    @click.argument('file_path')
    @click.option('--types', '-t', multiple=True, default=['all'], 
                  help='Analysis types to perform')
    @click.option('--deep', is_flag=True, help='Perform deep analysis')
    @click.option('--output', '-o', help='Output file for results')
    @click.pass_context
    def analyze(ctx, file_path, types, deep, output):
        """Analyze a file"""
        config = ctx.obj['config']
        formatter = ctx.obj['formatter']
        
        if not Path(file_path).exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return
        
        async def run_analysis():
            async with APIClient(config) as client:
                try:
                    with console.status("[bold green]Analyzing file...") as status:
                        request_data = {
                            "file_path": file_path,
                            "analysis_types": list(types),
                            "deep_scan": deep,
                            "include_metadata": True
                        }
                        
                        response = await client.post("/api/v1/analyze/file", json_data=request_data)
                        operation_id = response.get("operation_id")
                        
                        # Poll for results
                        while True:
                            result_response = await client.get(f"/api/v1/analyze/{operation_id}")
                            status_val = result_response.get("status")
                            
                            if status_val == "completed":
                                break
                            elif status_val == "failed":
                                console.print("[red]Analysis failed[/red]")
                                return
                            
                            await asyncio.sleep(1)
                    
                    # Display results
                    console.print("[bold green]Analysis completed![/bold green]")
                    formatted = formatter.format_data(result_response.get("results", {}))
                    
                    if isinstance(formatted, (Table, Tree)):
                        console.print(formatted)
                    else:
                        console.print(formatted)
                    
                    # Save to file if requested
                    if output:
                        with open(output, 'w') as f:
                            if output.endswith('.json'):
                                json.dump(result_response, f, indent=2, default=str)
                            else:
                                f.write(str(result_response))
                        console.print(f"[green]Results saved to: {output}[/green]")
                
                except Exception as e:
                    console.print(f"[red]Analysis error: {e}[/red]")
        
        asyncio.run(run_analysis())
    
    @cli.command()
    @click.pass_context
    def status(ctx):
        """Get system status"""
        config = ctx.obj['config']
        formatter = ctx.obj['formatter']
        
        async def get_status():
            async with APIClient(config) as client:
                try:
                    response = await client.get("/api/v1/status")
                    formatted = formatter.format_data(response)
                    
                    if isinstance(formatted, (Table, Tree)):
                        console.print(formatted)
                    else:
                        console.print(formatted)
                
                except Exception as e:
                    console.print(f"[red]Status error: {e}[/red]")
        
        asyncio.run(get_status())
    
    @cli.command()
    def shell():
        """Start interactive shell"""
        config = CLIConfig()
        shell = InteractiveShell(config)
        shell.cmdloop()
    
    @cli.command()
    @click.argument('key')
    @click.argument('value', required=False)
    @click.pass_context
    def config_cmd(ctx, key, value):
        """Get or set configuration values"""
        config = ctx.obj['config']
        
        if value is None:
            # Get value
            val = config.get(key)
            if val is not None:
                console.print(f"[bold]{key}[/bold]: {val}")
            else:
                console.print(f"[red]Configuration key '{key}' not found[/red]")
        else:
            # Set value
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                parsed_value = value
            
            config.set(key, parsed_value)
            console.print(f"[green]Set {key} = {parsed_value}[/green]")
    
    # Rename the config command to avoid conflict
    config_cmd.name = 'config'
    
    return cli


def main():
    """Main CLI entry point"""
    try:
        # Enable click completion
        click_completion_init()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="PhoenixDRS Professional CLI", add_help=False)
        parser.add_argument('--shell', action='store_true', help='Start interactive shell')
        parser.add_argument('--help', '-h', action='store_true', help='Show help')
        
        # Parse known args to check for shell mode
        known_args, remaining = parser.parse_known_args()
        
        if known_args.shell:
            # Start interactive shell
            config = CLIConfig()
            shell = InteractiveShell(config)
            shell.cmdloop()
        else:
            # Use Click CLI
            config = CLIConfig()
            cli = create_click_cli(config)
            
            # If no arguments provided or help requested, show help
            if not remaining or known_args.help:
                ctx = click.Context(cli)
                click.echo(cli.get_help(ctx))
                return
            
            # Execute Click command
            cli(remaining)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]CLI error: {e}[/red]")
        logger.exception("CLI error")


if __name__ == "__main__":
    main()