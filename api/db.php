<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "ottoman";

// Veritabanı bağlantısı
$conn = new mysqli($servername, $username, $password, $dbname);

// Bağlantı kontrolü
if ($conn->connect_error) {
    die("Veritabanı bağlantısı başarısız: " . $conn->connect_error);
}

// Arapça/Osmanlıca metnin doğru saklanması için UTF-8 (utf8mb4) karakter seti
$conn->set_charset("utf8mb4");
?>
