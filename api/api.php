<?php
require_once('db.php');

// Veri kaydetme: POST ile ottoman + turkish geldiğinde
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['ottoman']) && isset($_POST['turkish'])) {
    $ottoman = $_POST['ottoman'];
    $turkish = $_POST['turkish'];

    // Prepared statement ile SQL injection'a karşı güvenli ekleme
    $stmt = $conn->prepare("INSERT INTO translate (ottoman, turkish) VALUES (?, ?)");
    if ($stmt === false) {
        echo "error";
        $conn->close();
        exit();
    }
    $stmt->bind_param("ss", $ottoman, $turkish);

    if ($stmt->execute()) {
        echo "success"; // Başarılı ekleme işlemi
    } else {
        echo "error"; // Hata durumunda
    }

    $stmt->close();
    $conn->close();
    exit();
}

// Veri okuma: action=get_data
if (isset($_GET['action']) && $_GET['action'] == 'get_data') {
    $sql = "SELECT ottoman, turkish FROM translate";
    $result = $conn->query($sql);
    $data = array();

    if ($result && $result->num_rows > 0) {
        while ($row = $result->fetch_assoc()) {
            $entry = array(
                "ottoman_text" => $row["ottoman"],
                "turkish_translation" => $row["turkish"]
            );
            $data[] = $entry;
        }
        $total_records = $result->num_rows;
        echo json_encode(array("total_records" => $total_records, "data" => $data));
    } else {
        echo json_encode(array("total_records" => 0, "data" => array()));
    }

    $conn->close();
    exit();
}

echo "Invalid request"; // Geçersiz bir istek durumunda yanıt verin
?>
