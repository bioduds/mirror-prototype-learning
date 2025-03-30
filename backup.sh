#!/bin/bash

# Comprehensive backup script for mirror-prototype-learning project
# This script backs up all files including vectors and reports directories

# Exit on error
set -e

# Variables
BACKUP_DIR="$HOME/Documents/Mirrors/backups"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
PROJECT_NAME="mirror-prototype-learning"
BACKUP_FILE="${PROJECT_NAME}_${TIMESTAMP}.tar.gz"
REQUIRED_SPACE_MB=100  # Estimate required space in MB

# Function to display messages
log_message() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1"
}

# Function to handle errors
handle_error() {
    log_message "ERROR: $1"
    exit 1
}

# Function to check if directory exists, create if not
check_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        log_message "Creating directory: $dir"
        mkdir -p "$dir" || handle_error "Failed to create directory: $dir"
    else
        log_message "Directory already exists: $dir"
    fi
}

# Function to check available disk space
check_disk_space() {
    log_message "Checking available disk space..."
    local available_space_kb=$(df -k "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    local available_space_mb=$((available_space_kb / 1024))
    
    log_message "Available space: ${available_space_mb}MB, Required: ${REQUIRED_SPACE_MB}MB"
    
    if [ "$available_space_mb" -lt "$REQUIRED_SPACE_MB" ]; then
        handle_error "Not enough disk space. Available: ${available_space_mb}MB, Required: ${REQUIRED_SPACE_MB}MB"
    fi
}

# Function to create backup
create_backup() {
    log_message "Creating backup archive: $BACKUP_FILE"
    
    # Check if vectors and reports directories exist
    if [ ! -d "vectors" ] && [ ! -d "reports" ]; then
        log_message "Warning: Neither vectors nor reports directories found"
    elif [ ! -d "vectors" ]; then
        log_message "Warning: vectors directory not found"
    elif [ ! -d "reports" ]; then
        log_message "Warning: reports directory not found"
    fi
    
    # Create backup tarball
    tar -czvf "$BACKUP_FILE" --exclude='.git' $(git ls-files) vectors reports 2>/dev/null || handle_error "Failed to create backup archive"
    
    # Move backup to destination directory
    mv "$BACKUP_FILE" "$BACKUP_DIR/" || handle_error "Failed to move backup file to $BACKUP_DIR"
    
    log_message "Backup successfully created at: $BACKUP_DIR/$BACKUP_FILE"
}

# Main execution
main() {
    log_message "Starting backup process for $PROJECT_NAME..."
    
    # Check if backup directory exists
    check_directory "$BACKUP_DIR"
    
    # Check available disk space
    check_disk_space
    
    # Create backup
    create_backup
    
    log_message "Backup completed successfully."
}

# Run the main function
main

