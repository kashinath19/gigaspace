# Frontend Dockerfile
# Use an official Nginx image to serve the static files
FROM nginx:alpine

# Copy the build output to the Nginx directory
COPY ./dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80
