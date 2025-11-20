import os
from app import app, socketio

if __name__ == '__main__':
    # Set environment variables for production
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Run the application
    socketio.run(
        app,
        debug=debug,
        host='0.0.0.0',
        port=port,
        allow_unsafe_werkzeug=True  # For development only
    )