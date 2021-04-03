function parseDate(dateString) {
    let parts = dateString.split("-");
    return new Date(parts[0], parts[1] - 1, parts[2]);
}

function printDate(dateObject) {
    return `${dateObject.toLocaleDateString("en-US", {month: "long"})}, ${dateObject.toLocaleDateString("en-US", {year: "numeric"})}`;
}
