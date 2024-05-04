## EZ OCR Web Application
This is a web application that provides Optical Character Recognition (OCR) functionality. Users can upload images, and the application will extract text from those images using the VietOCR model. The extracted text is then stored in a MySQL database for future reference.
## Tech Stack
* Frontend: Bootstrap 5
* Backend: Node.js, Express.js
* Database: MySQL
* OCR Model: VietOCR
* Containerization: Docker
## Prerequisites
Node.js (v12 or later)
MySQL server
## Getting Started
1. Clone the repository:
```
git clone https://github.com/zet-rutherford/ezocr_datn.git
```
2. Install the dependencies:
```
cd web-ocr
npm install
```
3. Create a .env file in the root directory and add the following environment variables:
```
DB_HOST=your_mysql_host
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=your_mysql_database_name
```
4. Build and run the Docker container for the VietOCR model:
```
docker build -t vietocr .
docker run -d -p 8866:8866 vietocr
```
5. Start the Node.js server:
```
npm start
```
## Usage

* Navigate to http://localhost:3000 in your web browser.
* Login or create new account
* Click the "Upload Image" button and select an image file.
* The application will process the image using the VietOCR model and display the extracted text.
* The extracted text will also be stored in the MySQL database for future reference.