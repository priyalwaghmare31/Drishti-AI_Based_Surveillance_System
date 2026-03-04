// Camera stream management
let streamInterval;
let isStreamActive = false;

function startStream() {
    const cameraFrame = document.getElementById('cameraFrame');
    const statusIndicator = document.getElementById('statusIndicator');

    // Add cache buster to prevent caching
    const timestamp = Date.now();
    cameraFrame.src = `/video_feed?nocache=${timestamp}`;

    cameraFrame.onload = function() {
        statusIndicator.textContent = 'Online';
        statusIndicator.className = 'status-indicator status-online';
        isStreamActive = true;
    };

    cameraFrame.onerror = function() {
        statusIndicator.textContent = 'Offline';
        statusIndicator.className = 'status-indicator status-offline';
        isStreamActive = false;

        // Retry connection after 2 seconds
        setTimeout(startStream, 2000);
    };
}

function stopStream() {
    const cameraFrame = document.getElementById('cameraFrame');
    cameraFrame.src = '';
    isStreamActive = false;

    if (streamInterval) {
        clearInterval(streamInterval);
    }
}

// Auto-reconnect on page visibility change
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') {
        if (!isStreamActive) {
            startStream();
        }
    } else {
        // Optional: pause stream when tab is not visible
        // stopStream();
    }
});

// Start stream when page loads
document.addEventListener('DOMContentLoaded', function() {
    startStream();

    // Periodic health check every 30 seconds
    setInterval(function() {
        if (!isStreamActive) {
            startStream();
        }
    }, 30000);
});

// Ensure stream restarts on page show (for navigation back)
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible' && !isStreamActive) {
        startStream();
    }
});

// Handle page unload
window.addEventListener('beforeunload', function() {
    stopStream();
});
