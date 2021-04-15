"use strict";

const contentSelector = ".container.content";

function showError() {
    $(contentSelector).html(errorFragment());
}

function showProjects(data) {
    let container = $(contentSelector);
    if (data.hasOwnProperty("projects")) {
        data.projects.forEach(p => {
            container.append(
                `<div class="card-object flex-md-row flex-column">
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
                </div>`);
        });
    } else {
        showError();
    }
}

$(document).ready(function() {
    const url = "https://raw.githubusercontent.com/harshatech2012/harshatech2012.github.io/master/projects/project-list.json";
    fetch(url, {method: "GET"})
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                showError();
            }
        }).then(showProjects)
        .catch(error => {
            showError();
        });
});
