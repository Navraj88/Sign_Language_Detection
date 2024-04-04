function predictNumbers() {
    fetch('/set_labels_numbers')
        .then(response => response.json())
        .then(data => {
            console.log(data);
            document.getElementById('prediction').innerText = data.message;
        });
}

function predictAlphabets() {
    fetch('/set_labels_alphabets')
        .then(response => response.json())
        .then(data => {
            console.log(data);
            document.getElementById('prediction').innerText = data.message;
        });
}
