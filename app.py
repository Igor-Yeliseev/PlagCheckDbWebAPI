from app import create_app
from app.utils.db_utils import create_tables

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        # Create tables if they don't exist
        create_tables()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000) 