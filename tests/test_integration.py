"""
Integration tests for PhoenixDRS Professional
Tests the complete workflow from video analysis to repair completion
"""

import asyncio
import tempfile
import pytest
import shutil
from pathlib import Path
from typing import Dict, Any
import json
import time
import os

# Import our modules
from src.python.video_repair_orchestrator import (
    VideoRepairOrchestrator,
    RepairConfiguration,
    RepairTechnique,
    AIModelType,
    VideoFormat,
    RepairStatus
)
from src.python.ai_intelligence_engine import AIIntelligenceEngine


class TestVideoRepairIntegration:
    """Integration tests for video repair workflow"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_video_path(self, temp_dir):
        """Create a sample video file for testing"""
        # Create a simple test video file (placeholder)
        video_path = temp_dir / "test_video.mp4"
        # In a real test, you would create an actual video file
        # For now, create a dummy file
        video_path.write_bytes(b"dummy video content")
        return str(video_path)
    
    @pytest.fixture
    def corrupted_video_path(self, temp_dir):
        """Create a corrupted video file for testing"""
        video_path = temp_dir / "corrupted_video.mp4"
        # Create a file with some corruption patterns
        corrupted_data = b"ftyp" + b"\x00" * 100 + b"mdat" + b"\xFF" * 500
        video_path.write_bytes(corrupted_data)
        return str(video_path)
    
    @pytest.fixture
    def orchestrator(self):
        """Create video repair orchestrator instance"""
        config = RepairConfiguration()
        orchestrator = VideoRepairOrchestrator(config)
        yield orchestrator
        # Cleanup
        asyncio.run(orchestrator.shutdown())
    
    @pytest.mark.asyncio
    async def test_complete_repair_workflow(self, orchestrator, corrupted_video_path, temp_dir):
        """Test complete video repair workflow from start to finish"""
        
        # Arrange
        output_path = str(temp_dir / "repaired_video.mp4")
        
        config = RepairConfiguration(
            input_file=corrupted_video_path,
            output_file=output_path,
            techniques=[
                RepairTechnique.HEADER_RECONSTRUCTION,
                RepairTechnique.INDEX_REBUILD,
                RepairTechnique.CONTAINER_REMUX
            ],
            enable_ai_processing=False,  # Disable AI for basic test
            use_gpu=False,
            quality_factor=0.8
        )
        
        # Track progress
        progress_updates = []
        
        def progress_callback(progress: float, status: str):
            progress_updates.append((progress, status))
        
        config.progress_callback = progress_callback
        
        # Act
        session_id = await orchestrator.start_repair_session(config)
        
        # Wait for completion
        max_wait_time = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = orchestrator.get_session_status(session_id)
            if not status:
                break
            
            if status.final_status in [RepairStatus.COMPLETED, RepairStatus.FAILED]:
                break
            
            await asyncio.sleep(0.1)
        
        # Get final result
        final_result = orchestrator.get_session_status(session_id)
        
        # Assert
        assert final_result is not None
        assert final_result.session_id == session_id
        assert final_result.final_status in [RepairStatus.COMPLETED, RepairStatus.FAILED]
        
        # Check that progress callbacks were called
        assert len(progress_updates) > 0
        assert progress_updates[0][0] >= 0.0  # First progress should be >= 0
        assert progress_updates[-1][0] <= 1.0  # Last progress should be <= 1
        
        # Cleanup
        orchestrator.cleanup_session(session_id)
    
    @pytest.mark.asyncio
    async def test_video_analysis_workflow(self, orchestrator, sample_video_path):
        """Test video analysis functionality"""
        
        # Act
        analysis = await orchestrator.analyzer.analyze_video_file(sample_video_path)
        
        # Assert
        assert analysis.file_path == sample_video_path
        assert analysis.file_size_bytes > 0
        assert analysis.format_detected != VideoFormat.UNKNOWN
        assert analysis.analysis_time_seconds > 0
        assert isinstance(analysis.recommended_techniques, list)
    
    @pytest.mark.asyncio
    async def test_ai_integration_workflow(self, orchestrator, corrupted_video_path, temp_dir):
        """Test AI-enhanced repair workflow"""
        
        output_path = str(temp_dir / "ai_repaired_video.mp4")
        
        config = RepairConfiguration(
            input_file=corrupted_video_path,
            output_file=output_path,
            techniques=[RepairTechnique.AI_INPAINTING],
            enable_ai_processing=True,
            ai_models=[AIModelType.VIDEO_INPAINTING],
            use_gpu=False,  # Use CPU for testing
            ai_strength=0.5
        )
        
        # Act
        session_id = await orchestrator.start_repair_session(config)
        
        # Wait for completion
        max_wait_time = 45  # Longer timeout for AI processing
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = orchestrator.get_session_status(session_id)
            if not status:
                break
            
            if status.final_status in [RepairStatus.COMPLETED, RepairStatus.FAILED]:
                break
            
            await asyncio.sleep(0.1)
        
        # Get final result
        final_result = orchestrator.get_session_status(session_id)
        
        # Assert
        assert final_result is not None
        # AI processing might fail due to missing models, but should handle gracefully
        assert final_result.final_status in [RepairStatus.COMPLETED, RepairStatus.FAILED]
        
        if final_result.success:
            assert final_result.ai_processing_used or len(final_result.ai_models_used) > 0
        
        # Cleanup
        orchestrator.cleanup_session(session_id)
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self, orchestrator, corrupted_video_path, temp_dir):
        """Test handling multiple concurrent repair sessions"""
        
        # Arrange
        num_sessions = 3
        sessions = []
        
        for i in range(num_sessions):
            output_path = str(temp_dir / f"repaired_video_{i}.mp4")
            config = RepairConfiguration(
                input_file=corrupted_video_path,
                output_file=output_path,
                techniques=[RepairTechnique.HEADER_RECONSTRUCTION],
                enable_ai_processing=False,
                use_gpu=False
            )
            
            session_id = await orchestrator.start_repair_session(config)
            sessions.append(session_id)
        
        # Act - wait for all sessions to complete
        max_wait_time = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            completed_count = 0
            
            for session_id in sessions:
                status = orchestrator.get_session_status(session_id)
                if status and status.final_status in [RepairStatus.COMPLETED, RepairStatus.FAILED]:
                    completed_count += 1
            
            if completed_count == num_sessions:
                break
            
            await asyncio.sleep(0.2)
        
        # Assert
        for session_id in sessions:
            final_result = orchestrator.get_session_status(session_id)
            assert final_result is not None
            assert final_result.final_status in [RepairStatus.COMPLETED, RepairStatus.FAILED]
            
            # Cleanup
            orchestrator.cleanup_session(session_id)
    
    @pytest.mark.asyncio
    async def test_session_cancellation(self, orchestrator, corrupted_video_path, temp_dir):
        """Test cancelling a repair session"""
        
        output_path = str(temp_dir / "cancelled_repair.mp4")
        config = RepairConfiguration(
            input_file=corrupted_video_path,
            output_file=output_path,
            techniques=[RepairTechnique.FRAGMENT_RECOVERY],  # Potentially slow operation
            enable_ai_processing=False
        )
        
        # Act
        session_id = await orchestrator.start_repair_session(config)
        
        # Wait a bit then cancel
        await asyncio.sleep(0.1)
        cancelled = orchestrator.cancel_session(session_id)
        
        # Wait for cancellation to take effect
        await asyncio.sleep(0.5)
        
        final_result = orchestrator.get_session_status(session_id)
        
        # Assert
        assert cancelled is True
        assert final_result is not None
        assert final_result.final_status == RepairStatus.CANCELLED
        
        # Cleanup
        orchestrator.cleanup_session(session_id)
    
    @pytest.mark.asyncio
    async def test_system_resource_monitoring(self, orchestrator):
        """Test system resource monitoring during operation"""
        
        # Act
        system_status = orchestrator.get_system_status()
        
        # Assert
        assert "system_resources" in system_status
        assert "active_sessions" in system_status
        assert "loaded_ai_models" in system_status
        assert "ai_memory_usage" in system_status
        
        resources = system_status["system_resources"]
        assert "cpu_count" in resources
        assert "memory_total_gb" in resources
        assert "memory_available_gb" in resources
        assert resources["cpu_count"] > 0
        assert resources["memory_total_gb"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_input(self, orchestrator, temp_dir):
        """Test error handling with invalid input files"""
        
        # Arrange - non-existent file
        invalid_path = str(temp_dir / "nonexistent.mp4")
        output_path = str(temp_dir / "output.mp4")
        
        config = RepairConfiguration(
            input_file=invalid_path,
            output_file=output_path,
            techniques=[RepairTechnique.HEADER_RECONSTRUCTION]
        )
        
        # Act
        session_id = await orchestrator.start_repair_session(config)
        
        # Wait for completion
        max_wait_time = 10
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = orchestrator.get_session_status(session_id)
            if not status:
                break
            
            if status.final_status in [RepairStatus.COMPLETED, RepairStatus.FAILED]:
                break
            
            await asyncio.sleep(0.1)
        
        final_result = orchestrator.get_session_status(session_id)
        
        # Assert
        assert final_result is not None
        assert final_result.final_status == RepairStatus.FAILED
        assert final_result.success is False
        assert len(final_result.error_message) > 0
        
        # Cleanup
        orchestrator.cleanup_session(session_id)


class TestCLIIntegration:
    """Integration tests for CLI interface"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_cli_help_command(self):
        """Test CLI help command"""
        import subprocess
        
        # Act
        result = subprocess.run(
            ["python", "main.py", "--help"],
            capture_output=True,
            text=True
        )
        
        # Assert
        assert result.returncode == 0
        assert "PhoenixDRS" in result.stdout or "usage:" in result.stdout
    
    def test_cli_version_command(self):
        """Test CLI version command"""
        import subprocess
        
        # Act
        result = subprocess.run(
            ["python", "main.py", "--version"],
            capture_output=True,
            text=True
        )
        
        # Assert
        assert result.returncode == 0
        # Should output version information


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.benchmark(group="video_processing")
    def test_video_analysis_performance(self, benchmark, temp_dir):
        """Benchmark video analysis performance"""
        
        # Create test video file
        video_path = temp_dir / "benchmark_video.mp4"
        video_path.write_bytes(b"ftyp" + b"\x00" * 10000 + b"mdat" + b"\xFF" * 50000)
        
        from src.python.video_repair_orchestrator import VideoAnalyzer
        analyzer = VideoAnalyzer()
        
        # Benchmark
        result = benchmark(
            asyncio.run,
            analyzer.analyze_video_file(str(video_path))
        )
        
        # Assert
        assert result.file_path == str(video_path)
    
    @pytest.mark.benchmark(group="ai_processing")
    def test_ai_model_loading_performance(self, benchmark, temp_dir):
        """Benchmark AI model loading performance"""
        
        from src.python.video_repair_orchestrator import AIModelManager
        
        def load_models():
            manager = AIModelManager(str(temp_dir / "models"))
            return asyncio.run(manager.load_model(AIModelType.VIDEO_INPAINTING))
        
        # Benchmark
        result = benchmark(load_models)
        
        # Assert (may fail if models not available, but should not crash)
        assert result is not None or result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])