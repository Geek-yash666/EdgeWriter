#!/usr/bin/env python3
"""
EdgeWriter Server - Forces Chrome/Edge to use NVIDIA GPU for WebGPU acceleration
Integrates GPU detection with forced high-performance GPU browser launch
"""

import http.server
import socketserver
import subprocess
import tempfile
import shutil
import atexit
import sys
import json
import os
import psutil
import time
import threading

PORT = 8000
URL = f"http://localhost:{PORT}"

# Create temporary Chrome profile (so flags don't affect your main Chrome)
temp_profile = tempfile.mkdtemp(prefix="edgewriter_gpu_force_")
browser_process = None
server_running = True
httpd_ref = None  # set in main so monitor can request shutdown
cleanup_done = False

def cleanup():
    """Clean up temporary browser profile and browser process on exit"""
    global browser_process, cleanup_done
    if cleanup_done:
        return
    cleanup_done = True
    
    try:
        # Kill browser process if it's still running
        if browser_process and browser_process.poll() is None:
            print(f"Closing browser...")
            browser_process.terminate()
            try:
                browser_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                browser_process.kill()
        
        # Clean up temp profile
        shutil.rmtree(temp_profile, ignore_errors=True)
        print(f"Cleaned up temporary profile")
    except Exception as e:
        print(f"Warning: Could not clean up: {e}")

atexit.register(cleanup)

def get_gpu_info():
    """Detect available GPUs on the system"""
    gpus = []
    try:
        # nvidia-smi for NVIDIA GPUs
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                encoding='utf-8',
                stderr=subprocess.DEVNULL
            )
            for line in output.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 1:
                    gpus.append({
                        "name": parts[0].strip(),
                        "type": "NVIDIA",
                        "memory": parts[1].strip() if len(parts) > 1 else "Unknown"
                    })
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Fallback to WMIC on Windows for all GPUs
        if sys.platform == 'win32':
            try:
                output = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "Name"],
                    encoding='utf-8',
                    stderr=subprocess.DEVNULL
                )
                lines = output.strip().split('\n')
                for line in lines:
                    name = line.strip()
                    if name and "Name" not in name and not any(g['name'] == name for g in gpus):
                        gpu_type = "Integrated" if any(x in name.upper() for x in ["INTEL", "AMD RADEON(TM) GRAPHICS"]) else "Dedicated"
                        gpus.append({"name": name, "type": gpu_type, "memory": "Unknown"})
            except Exception:
                pass

    except Exception as e:
        print(f"Error detecting GPUs: {e}")

    return gpus


def get_system_info():
    """Return basic system info including RAM in GB."""
    ram_gb = None
    try:
        mem = psutil.virtual_memory()
        if mem and mem.total:
            ram_gb = round(mem.total / (1024 ** 3))
    except Exception:
        pass

    # Fallback to WMIC if psutil not sufficient
    if ram_gb is None and sys.platform == 'win32':
        try:
            output = subprocess.check_output(
                ["wmic", "OS", "get", "TotalVisibleMemorySize"],
                encoding='utf-8',
                stderr=subprocess.DEVNULL
            )
            lines = [ln.strip() for ln in output.splitlines() if ln.strip() and "TotalVisibleMemorySize" not in ln]
            if lines:
                kb = float(lines[0])
                ram_gb = round((kb * 1024) / (1024 ** 3))
        except Exception:
            pass

    return {"ramGB": ram_gb}

def find_browser():
    """Find Chrome or Edge executable"""
    if sys.platform == 'win32':
        # Common Chrome/Edge locations on Windows
        possible_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        try:
            subprocess.check_output(["where", "chrome.exe"], stderr=subprocess.DEVNULL)
            return "chrome.exe"
        except:
            pass
        try:
            subprocess.check_output(["where", "msedge.exe"], stderr=subprocess.DEVNULL)
            return "msedge.exe"
        except:
            pass
    else:
        # Linux/Mac
        for browser in ["google-chrome", "chrome", "chromium", "microsoft-edge"]:
            try:
                subprocess.check_output(["which", browser], stderr=subprocess.DEVNULL)
                return browser
            except:
                pass
    return None

class Handler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with GPU info API endpoint"""
    
    def log_message(self, format, *args):
        """Silent logging for cleaner output"""
        pass
    
    def do_GET(self):
        if self.path == '/api/gpu-info':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            gpus = get_gpu_info()
            system = get_system_info()
            self.wfile.write(json.dumps({"gpus": gpus, **system}).encode('utf-8'))
        else:
            super().do_GET()

def launch_browser_with_gpu_force(browser_path):
    global browser_process
    browser_flags = [
        "--new-window",
        f"--user-data-dir={temp_profile}",
        "--force-high-performance-gpu",              # Force dGPU over iGPU
        #"--enable-unsafe-webgpu",                    # Enable full WebGPU
        "--ignore-gpu-blocklist",                    # Don't block older GTX cards
        "--enable-features=Vulkan",                  # Enable Vulkan backend
        "--disable-features=UseChromeOSDirectVideoDecoder",
        "--no-first-run",
        "--no-default-browser-check",
        #"--disable-gpu-sandbox",                     # Helps on some systems
        "--disable-software-rasterizer",             # Force hardware acceleration
        "--disable-background-mode",                 #  Prevent Chrome from staying in background
        "--disable-background-networking",           # Prevent background processes
        "--disable-sync",                            # Disable sync to prevent background activity
        "--disable-extensions",                      # Disable extensions that might keep it alive
        "--no-service-autorun",                      # Don't auto-run services
        URL
    ]
    
    cmd = [browser_path, *browser_flags]
    
    try:
        browser_process = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        print(f"Error launching browser: {e}")
        return False

def monitor_browser():
    """Monitor browser process and shutdown server when browser closes"""
    global browser_process, server_running, httpd_ref
    
    if not browser_process:
        return
    
    print("  Monitoring browser process (server will stop when browser closes)...")
    
    # Wait for browser to close
    try:
        browser_process.wait()
        
        # Additional check: ensure all Chrome processes from this profile are gone
        time.sleep(0.5)  # Brief delay to let processes clean up
        
        # Kill any remaining Chrome processes using our temp profile
        if sys.platform == 'win32':
            try:
                # Find processes with our temp profile path
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] and 'chrome' in proc.info['name'].lower():
                            cmdline = proc.info.get('cmdline', [])
                            if cmdline and temp_profile in ' '.join(cmdline):
                                proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception as e:
                print(f"Warning: Could not clean up remaining processes: {e}")
        
        print("\n" + "─" * 70)
        print("Browser closed - shutting down server...")
        print("─" * 70)
        server_running = False
        if httpd_ref:
            try:
                httpd_ref.shutdown()
            except Exception as e:
                print(f"Warning: server shutdown request failed: {e}")
        cleanup()
        return
    except Exception as e:
        print(f"Browser monitoring error: {e}")

def main():
    """Main server setup and launch"""
    print("=" * 70)
    print("EdgeWriter - On-Device AI Writing Assistant")
    print("=" * 70)
    print(f"\nDetecting GPUs...")
    
    system_gpus = get_gpu_info()
    has_nvidia = False
    
    if system_gpus:
        for gpu in system_gpus:
            print(f"  ✓ Found: {gpu['name']}")
            if gpu.get('memory'):
                print(f"    Memory: {gpu['memory']}")
            if gpu['type'] == 'NVIDIA':
                has_nvidia = True
    else:
        print("  ⚠ No GPUs detected via system tools")
    
    print(f"\nStarting server at {URL}")
    
    # Try to launch browser with GPU forcing
    browser_path = find_browser()
    
    if browser_path:
        print(f"\nLaunching browser with NVIDIA GPU acceleration...")
        print(f"  Browser: {os.path.basename(browser_path)}")
        print(f"  Profile: {temp_profile}")
        print(f"  Flags: ForceHighPerformanceGPU, WebGPU enabled")
        
        if launch_browser_with_gpu_force(browser_path):
            print(f"\n✓ Browser launched successfully!")
            if has_nvidia:
                print(f"  → Expected performance: 50-100+ tokens/sec on NVIDIA GPU")
            else:
                print(f"  ⚠ No NVIDIA GPU detected - will use best available GPU")
            
            # Start browser monitoring in background thread
            monitor_thread = threading.Thread(target=monitor_browser, daemon=True)
            monitor_thread.start()
        else:
            print(f"\n⚠ Could not launch browser automatically")
            print(f"  Please open manually: {URL}")
    else:
        print(f"\n⚠ Chrome/Edge not found in standard locations")
        print(f"\nManual launch command:")
        print(f'  chrome.exe --enable-features=ForceHighPerformanceGPU "{URL}"')
        print(f"\nOr simply navigate to: {URL}")
    
    if not has_nvidia:
        print("\n" + "─" * 70)
        print("GPU Optimization Tips:")
        print("─" * 70)
        print("If you have an NVIDIA GPU but it's not detected:")
        print("  1. Install/update NVIDIA drivers")
        print("  2. Restart your computer")
        print("  3. Re-run this script")
        print("\nIf browser uses integrated GPU instead of NVIDIA:")
        print("  → Visit chrome://flags/#force-high-performance-gpu")
        print("     Set to 'Enabled' and relaunch browser")
        print("─" * 70)
    
    print(f"\n{'─' * 70}")
    print(f"Server running on http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop (this will also close the browser)")
    print(f"{'─' * 70}\n")
    
    # Start HTTP server
    global httpd_ref
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd_ref = httpd
        try:
            httpd.serve_forever()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            print("\n\n" + "─" * 70)
            print("Shutting down server and browser...")
            print("─" * 70)
            cleanup()
            print("Server stopped. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()