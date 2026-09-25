<?php
// Eski surum $_POST degerlerini SQL'e ham string olarak gomuyordu (SQL
// injection). api.php ile ayni prepared-statement deseni ve tek db.php
// baglantisi kullanilir.
require_once('db.php');

if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['ottoman']) && isset($_POST['turkish'])) {
    $ottoman = $_POST['ottoman'];
    $turkish = $_POST['turkish'];

    if (empty($ottoman) || empty($turkish)) {
        echo "empty";
        $conn->close();
        exit();
    }

    $stmt = $conn->prepare("INSERT INTO translate (ottoman, turkish) VALUES (?, ?)");
    if ($stmt === false) {
        echo "error";
        $conn->close();
        exit();
    }
    $stmt->bind_param("ss", $ottoman, $turkish);

    if ($stmt->execute()) {
        echo "success";
    } else {
        echo "error";
    }

    $stmt->close();
    $conn->close();
    exit();
}

// Get Data: JSON olarak indirme
if (isset($_GET['action']) && $_GET['action'] == 'get_data') {
    $sql = "SELECT ottoman, turkish FROM translate";
    $result = $conn->query($sql);
    $data = array();

    if ($result && $result->num_rows > 0) {
        while ($row = $result->fetch_assoc()) {
            $data[] = array(
                "ottoman_text" => $row["ottoman"],
                "turkish_translation" => $row["turkish"]
            );
        }
    }

    $conn->close();

    header("Content-Type: application/json");
    header("Content-Disposition: attachment; filename=data.json");
    echo json_encode(array("data" => $data), JSON_UNESCAPED_UNICODE);
    exit();
}

echo "Invalid request";
?>
