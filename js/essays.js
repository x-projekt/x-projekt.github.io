"use strict";

function showProjects(data) {
    let container = $(".container.essays-insert");
    data.essays.sort(function(e1, e2) {
        // lastest first, sort in descending order
        // e2 - e1 < 0 => e1 comes before e2
        return parseDate(e2["card.date"]) - parseDate(e1["card.date"]);
    })

    if (data.hasOwnProperty("essays")) {
        data.essays.forEach(p => {
            container.append(
                `<div class="card-object flex-md-row flex-column">
                    <div class="card-content">
                        <div class="card-heading">
                            <a href="${p["card.title"]["href"]}">
                                ${p["card.title"]["text"]}
                            </a>
                        </div>
                        <p class="lead">
                            ${p["card.gist"]}
                        </p>
                        <hr class="w-100 my-0">
                        <div class="card-footing">
                            <div class="d-inline-flex flex-wrap w-auto">
                                ${p["card.tags"].map(function (key) {
                                    return `<div class="hashtags">#${key}</div>`
                                }).join("\n")}
                            </div>
                            <div class="card-date">${printDate(parseDate(p["card.date"]))}</div>
                        </div>
                    </div>
                </div>`);
        });
    } else {
        insertError("Missing JSON property: 'projects' property not found in project-list.json");
    }
}

$(document).ready(function() {
    const url = "../essays/essay-list.json";
    fetch(url, {method: "GET"})
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                insertError("Unable to fetch project-list.json");
            }
        }).then(showProjects)
        .catch(error => {
            insertError(error);
        });
});
