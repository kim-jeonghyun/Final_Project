let gender = 'w'
let category = 'top'
let s3_url = "";

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

change_to_paired_model = function (gender, category, item_id) {
    console.log("change_to_paired_model");
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function () {
        document.getElementById("model").innerHTML = this.responseText;
    }
    xhttp.open("GET", "/click_item2?item_id=" + item_id + "&category=" + category + '&gender=' + gender);
    xhttp.send();
}

change_to_generated_model = function (s3_url, item_id) {
    console.log("change_to_generated_model");
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function () {
        text = '<img class="img-fluid rounded" src='+this.responseText+'>';
        document.getElementById("model").innerHTML = text;
    }
    xhttp.open("GET", "/click_item?item_id=" + item_id + "&category=" + category + "&s3_url=" + s3_url + '&gender=' + gender);
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
    if (s3_url == ""){
        change_to_paired_model(gender, category, item_id);
    } else{
        change_to_generated_model(s3_url, item_id);
    }
}

upload_image = async function (element) {
    let formData = new FormData();
    formData.append("file", element.files[0]);
    await fetch('/upload', {
        method: "POST",
        body: formData
    }).then(function (response) {
        return response.text().then(function (url) {
            s3_url=url;
            text = '<img class="img-fluid rounded" src='+url+'>';
            document.getElementById("model").innerHTML = text;
        });
    });
}