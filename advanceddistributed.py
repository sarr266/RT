# advanced_distributed.py
from sys import exit
import pygame as pg
import ray
import numpy as np
import time
import os

from objects import Sphere, Checkerboard
from camera import Camera
from skybox import Skybox
from vector import Vector
from light import Light
from ray_ import Ray

# Configuration options
screen_size = Vector(1280, 720)
shadow_bias = 0.0001
max_reflections = 3
num_workers = os.cpu_count() or 4  # Use all available CPU cores

# Scene setup functions
def create_scene():
    camera = Camera(Vector(0, 0, 0), screen_size, 90)
    skybox = Skybox("skybox.png")
    
    objects = [
        Sphere(Vector(0, 0, -10), 2, Vector(1, 0, 0)),
        Sphere(Vector(5, 0, -15), 2, Vector(0, 1, 0)),
        Sphere(Vector(-5, 0, -15), 2, Vector(0, 0, 1)),
        Checkerboard(2, Vector(0, 0, 0), Vector(1, 1, 1))
    ]
    
    light = Light(Vector(-1, 1, -1), 1)
    
    return camera, skybox, objects, light

# Ray worker class - persistent worker that processes chunks of the image
@ray.remote
class RayTracingWorker:
    def __init__(self, skybox, objects, light, shadow_bias, max_reflections):
        self.skybox = skybox
        self.objects = objects
        self.light = light
        self.shadow_bias = shadow_bias
        self.max_reflections = max_reflections
    
    def trace_chunk(self, start_x, start_y, width, height, camera):
        """Trace a rectangular chunk of pixels"""
        results = []
        
        for y in range(start_y, start_y + height):
            for x in range(start_x, start_x + width):
                if x >= screen_size.x or y >= screen_size.y:
                    continue
                    
                ray = camera.get_direction(Vector(x, y))
                color = self.trace_ray_with_reflections(ray)
                results.append((x, y, color.to_rgb()))
                
        return results
    
    def trace_ray_with_reflections(self, ray):
        """Trace a ray with reflections"""
        color, intersect, normal = self.trace_ray(ray)
        
        if intersect:
            reflection_ray = Ray(intersect + ray.direction.reflect(normal) * self.shadow_bias, 
                               ray.direction.reflect(normal))
            reflection_color = Vector(0, 0, 0)
            reflection_times = 0
            
            for reflection in range(self.max_reflections):
                new_color, new_intersect, new_normal = self.trace_ray(reflection_ray)
                reflection_color += new_color
                reflection_times += 1
                
                if new_intersect:
                    reflection_ray = Ray(new_intersect + reflection_ray.direction.reflect(new_normal) * self.shadow_bias, 
                                       reflection_ray.direction.reflect(new_normal))
                else:
                    break
                    
            if reflection_times > 0:
                color += reflection_color / reflection_times
        else:
            color = self.skybox.get_image_coords(ray.direction)
            
        return color
        
    def trace_ray(self, ray):
        """Basic ray tracing function"""
        color = Vector(0, 0, 0)
        intersect, object = ray.cast(self.objects)
        normal = False
        
        if intersect:
            normal = object.get_normal(intersect)
            color = object.get_color(intersect)
            light_ray = Ray(intersect + normal * self.shadow_bias, -self.light.direction.normalize())
            light_intersect, obstacle = light_ray.cast(self.objects)
            
            if light_intersect:
                color *= 0.1 / self.light.strength
                
            color *= normal.dot(-self.light.direction * self.light.strength)
        else:
            color = self.skybox.get_image_coords(ray.direction)
            
        return color, intersect, normal

def main():
    # Initialize Ray with performance monitoring
    ray.init(include_dashboard=True)
    
    pg.init()
    pg.display.init()

    display = pg.display.set_mode((screen_size.x, screen_size.y))
    pg.display.set_caption("Advanced Distributed Raytracer")
    
    # Create scene objects
    camera, skybox, objects, light = create_scene()
    
    # Create worker actors
    workers = [RayTracingWorker.remote(skybox, objects, light, shadow_bias, max_reflections) 
              for _ in range(num_workers)]
    
    # Create a blank surface to render to
    render_surface = pg.Surface((screen_size.x, screen_size.y))
    pixel_array = pg.PixelArray(render_surface)
    
    # Configure chunk size based on screen dimensions
    # Balance: too small = more overhead, too large = less parallelism
    chunk_size = 64
    
    # Start timing
    start_time = time.time()
    
    # Create rendering tasks
    tasks = []
    for y in range(0, screen_size.y, chunk_size):
        for x in range(0, screen_size.x, chunk_size):
            # Distribute chunks across workers in a round-robin fashion
            worker_idx = (x // chunk_size + y // chunk_size) % len(workers)
            width = min(chunk_size, screen_size.x - x)
            height = min(chunk_size, screen_size.y - y)
            tasks.append(workers[worker_idx].trace_chunk.remote(x, y, width, height, camera))
    
    # Process results
    completed = 0
    total_chunks = len(tasks)
    
    # Message to display
    font = pg.font.SysFont(None, 36)
    
    while tasks:
        # Process a batch of completed tasks
        done_ids, tasks = ray.wait(tasks, num_returns=1)
        
        # Get and process results
        for chunk_results in ray.get(done_ids):
            for x, y, color in chunk_results:
                pixel_array[x, y] = color
            
            # Update completion count
            completed += 1
            
        # Display current progress
        pg.surfarray.blit_array(display, pg.surfarray.array3d(render_surface))
        
        # Draw progress percentage
        progress_text = f"Rendering: {completed}/{total_chunks} chunks ({int(completed/total_chunks*100)}%)"
        text_surface = font.render(progress_text, True, (255, 255, 255))
        display.blit(text_surface, (10, 10))
        
        pg.display.flip()
        
        # Check for quit events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                ray.shutdown()
                pg.quit()
                exit()
    
    # Free the pixel array
    pixel_array.close()
    
    end_time = time.time()
    render_time = end_time - start_time
    print(f"Rendering completed in {render_time:.2f} seconds")
    
    # Display final render time on the image
    final_text = f"Render time: {render_time:.2f}s using {num_workers} workers"
    text_surface = font.render(final_text, True, (255, 255, 255))
    display.blit(text_surface, (10, 10))
    pg.display.flip()

    # Main event loop after rendering is complete
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                ray.shutdown()
                pg.quit()
                exit()

if __name__ == "__main__":
    main()