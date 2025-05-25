from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import asyncio

from .base_agent import BaseAgent
from ..core.config import SystemConfig
from ..core.interfaces.video_generator import IVideoGenerator

class VideoGeneratorAgent(BaseAgent):
    """Agent responsible for generating and processing videos."""
    
    def __init__(self, 
                 agent_id: str,
                 config: SystemConfig,
                 video_generator: IVideoGenerator,
                 expertise: Optional[Dict[str, float]] = None):
        super().__init__(
            agent_id=agent_id,
            role="video_generator",
            config=config,
            expertise=expertise or {
                'video_generation': 0.9,
                'visual_effects': 0.85,
                'content_optimization': 0.8
            }
        )
        self.video_generator = video_generator
        self.generation_queue = asyncio.Queue()
        self.processing_tasks = set()
    
    async def initialize(self) -> None:
        """Initialize the video generator agent."""
        await super().initialize()
        # Start the background processing task
        self.processing_tasks.add(
            asyncio.create_task(self._process_generation_queue())
        )
    
    async def shutdown(self) -> None:
        """Clean up resources and stop processing."""
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        self.processing_tasks.clear()
        
        await super().shutdown()
    
    async def _execute_task_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement video generation task execution."""
        try:
            # Validate input
            if not self.video_generator.validate_input(task):
                return {
                    'success': False,
                    'error': 'Invalid input content',
                    'task_type': 'video_generation'
                }
            
            # Add task to generation queue
            await self.generation_queue.put(task)
            
            return {
                'success': True,
                'message': 'Task queued for processing',
                'task_id': task.get('task_id', ''),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in video generation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_type': 'video_generation'
            }
    
    async def _process_generation_queue(self) -> None:
        """Process tasks from the generation queue."""
        while True:
            try:
                # Get task from queue
                task = await self.generation_queue.get()
                
                # Generate video
                result = await self._generate_video(task)
                
                # Process result
                await self._handle_generation_result(result)
                
                # Mark task as done
                self.generation_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing generation queue: {str(e)}")
    
    async def _generate_video(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video from task content."""
        try:
            # Extract generation parameters
            input_content = task.get('content', {})
            user_preferences = task.get('preferences', {})
            
            # Generate video
            result = await self.video_generator.generate_video(
                input_content=input_content,
                user_preferences=user_preferences
            )
            
            return {
                'success': True,
                'result': result,
                'task_id': task.get('task_id', ''),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating video: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task.get('task_id', '')
            }
    
    async def _handle_generation_result(self, result: Dict[str, Any]) -> None:
        """Handle the result of video generation."""
        if result['success']:
            # Store metrics
            self.metrics['total_generations'] += 1
            self.metrics['successful_generations'] += 1
            
            # Get generation metrics
            metrics = await self.video_generator.get_generation_metrics()
            self.metrics['generation_metrics'] = metrics
            
        else:
            self.metrics['total_generations'] += 1
            self.metrics['failed_generations'] += 1
    
    async def optimize_video(self, video_id: str, optimization_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize an existing video."""
        try:
            result = await self.video_generator.optimize_video(
                video_id=video_id,
                optimization_params=optimization_params
            )
            return {
                'success': True,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error optimizing video: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def export_video(self, video_id: str, format: str, quality: str) -> Dict[str, Any]:
        """Export video in specified format and quality."""
        try:
            result = await self.video_generator.export_video(
                video_id=video_id,
                format=format,
                quality=quality
            )
            return {
                'success': True,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error exporting video: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            } 