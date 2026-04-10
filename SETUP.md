# Fake Review Detector - Authentication Setup Guide

## Prerequisites

1. **Python 3.7+** installed on your system
2. **MongoDB** running on localhost:27017
3. **pip** package manager

## Installation Steps
py -V:3.13 app.py

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MongoDB

Make sure MongoDB is running on your local machine:

**Windows:**
```bash
# Start MongoDB service
net start MongoDB
```

**macOS/Linux:**
```bash
# Start MongoDB service
sudo systemctl start mongod
# or
mongod
```

### 3. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Default Admin Account

A default admin account is automatically created on first run:

- **Username:** `admin`
- **Password:** `admin123`

**Important:** Change the default admin password in production!

## Features Added

### 1. User Authentication
- User registration and login
- Secure password hashing with bcrypt
- Session management with Flask-Login

### 2. Admin Features
- Admin login and dashboard
- User management (view and delete users)
- Admin-only access controls

### 3. Database Integration
- MongoDB connection for user data storage
- Separate collections for users and admins
- Automatic admin account creation

## Usage

### For Regular Users:
1. Visit `http://localhost:5000`
2. Click "Register" to create an account
3. Login with your credentials
4. Use the fake review detection features

### For Admins:
1. Login with admin credentials (admin/admin123)
2. Access the Admin Dashboard
3. Manage users and view system statistics

## Security Notes

- Passwords are hashed using bcrypt
- Session management is handled securely
- Admin routes are protected with authentication
- Default admin password should be changed in production

## Database Collections

- `users`: Stores regular user accounts
- `admins`: Stores admin accounts

## Troubleshooting

### MongoDB Connection Issues:
- Ensure MongoDB is running on localhost:27017
- Check if MongoDB service is started
- Verify connection string in app.py

### Import Errors:
- Run `pip install -r requirements.txt` to install all dependencies
- Ensure Python version is 3.7 or higher

### Authentication Issues:
- Clear browser cookies if experiencing login problems
- Check if user exists in MongoDB collections
- Verify password hashing is working correctly
