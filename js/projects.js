"use strict";

let json = `{
    "placeholders": {
        "title": "{project.title}"
    },

    "projects": [
        {
            "card.image": {
                "data": "./img/svg/sensor.svg"
            },
            "card.title": {
                "text": "Dual Raspberry Pi Stereo Vision",
                "href": "./projects/dual-raspi-stereo-vision.html"
            },
            "card.description": "A stereo-vision system composed of two Raspberry Pi's each with an infrared camera module. The images are processed using the OpenCV library while a custom software is used for synchronizing the two cameras.",
            "card.tags": ["python", "computer-vision"],
            "card.date": {
                "to": "2016-06-20",
                "from": "2017-04-20"
            }
        },
        {
            "card.image": {
                "data": "./img/svg/visual-effects.svg"
            },
            "card.title": {
                "text": "Product design and visual effects",
                "href": "./projects/product-design-vfx.html"
            },
            "card.description": "This details the products I designed and their associated product design videos I created.",
            "card.tags": ["solidworks", "after-effects", "3ds-max", "illustrator"],
            "card.date": {
                "to": "2017-05-20",
                "from": "2013-03-20"
            }
        }
    ]
}`;

let projects = JSON.parse(json)["projects"];
let p;
let projectList = "";

for (let i = 0; i < projects.length; i++) {
    p = projects[i];
    projectList += `<div class="card-object flex-md-row flex-column">
            <object class="card-image" type="image/svg+xml" data="${p["card.image"]["data"]}">
                <img src="./img/fallback-image.png" height="200" width="200">
            </object>
            <div class="card-content">
                <div class="card-heading">
                    <a href="${p["card.title"]["href"]}">
                        ${p["card.title"]["text"]}
                    </a>
                </div>
                <p class="lead">
                    ${p["card.description"]}
                </p>
                <hr class="w-100 my-0">
                <div class="card-footing">
                    <div class="d-inline-flex flex-wrap w-auto">
                        ${p["card.tags"].map(function (key) {
                            return `<div class="hashtags">#${key}</div>`
                        }).join("\n")}
                    </div>
                    <div class="card-date">${printDate(parseDate(p["card.date"]["from"]))} - ${printDate(parseDate(p["card.date"]["to"]))}</div>
                </div>
            </div>
        </div>`;
}
document.querySelector(".container.content").innerHTML += projectList;
