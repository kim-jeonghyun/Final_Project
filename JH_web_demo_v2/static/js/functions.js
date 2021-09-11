let gender = 'w'
let category = 'top'

change_model = function (gender, category) {
    console.log("change_model");
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function () {
        document.getElementById("model").innerHTML = this.responseText;
    }
    xhttp.open("GET", "/click_gender?gender=" + gender + "&category" + category);
    xhttp.send();
}

change_items = function (gender, category) {
        console.log("change_items");
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function () {
        document.getElementById("items").innerHTML = this.responseText;
    }
    xhttp.open("GET", "/click_category?category=" + category + "&gender=" + gender);
    xhttp.send();
}

change_to_generated_model = function (gender, item_id) {
    console.log("change_to_generated_model");
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function () {
        document.getElementById("model").innerHTML = this.responseText;
    }
    xhttp.open("GET", "/click_item?item_id=" + item_id + "&gender=" + gender);
    xhttp.send();
}

click_gender = function (element) {
    console.log("click_gender");
    const gender_new = element.getAttribute("id");
    console.log(gender_new);
    if (gender == gender_new) {
        change_model(gender, category);
    } else {
        gender = gender_new;
        change_model(gender, category);
        change_items(gender, category);
    }
}

click_category = function (element) {
    console.log("click_category");
    category = element.getAttribute("id");
    change_items(gender, category);
}

click_item = function (element) {
    console.log("click_item");
    item_id = element.getAttribute("id");
    change_to_generated_model(gender, item_id);
}

upload_image = async function (element) {
    let formData = new FormData();
    formData.append("file", element.files[0]);
    await fetch('/upload', {
        method: "POST",
        body: formData
    });
}