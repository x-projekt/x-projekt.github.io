function parseDate(dateString) {
    let parts = dateString.split("-");
    return new Date(parts[0], parts[1] - 1, parts[2]);
}

function printDate(dateObject) {
    return `${dateObject.toLocaleDateString(
        "en-US", {month: "long"})}, ${dateObject.toLocaleDateString("en-US", {year: "numeric"})}`;
}

function insertError(error) {
	let snippetDOM = null;
	$.get("../html/error.html", function(htmlCode) {
		snippetDOM = $($.parseHTML(htmlCode));
		snippetDOM.find(".error-message-insert").append(error);
		$(".error-insert").html(snippetDOM.prop("outerHTML"));
	});
}

const rootPages = {
	"about": "about.html",
	"projects": "projects.html",
	"publications": "publications.html",
	"essays": "essays.html",
	"contact": "contact.html",
	"home": "index.html",
	"credits": "credits.html",
};

const depthPrefix = "../";
const error = 404;
const affix = "$";

function getURL(urlId, depth) {
	if (!rootPages.hasOwnProperty(urlId)) {
		throw `Unknown href URL-ID: ${urlId}. [${location.pathname.split("/").pop()}]`;
	}

	var intDepth = parseInt(depth) || -1;
	if (intDepth === -1) {
		throw `Invalid document-depth: ${depth}. [${location.pathname.split("/").pop()}]`;
	}

	return depthPrefix.repeat(parseInt(depth)).concat(rootPages[urlId]);
}

function updateURLs(dom, bodyData) {
	let href = null, url = null;
	dom.find("a").each(function () {
		href = $(this).attr("href");
		if (href.indexOf(affix) === 0 && href.lastIndexOf(affix) === href.length - 1 
				&& href.length >= 2) { // what if href = "$"? hence, the length check.
			try {
				url = getURL(href.substr(1, href.length - 2), bodyData["depth"]);
				$(this).attr("href", url);
			} catch (error) {
				insertError(error);
			}
		}
	});
}

function insertHeader(headerURL) {
	let headerContainer = $("body > .header-insert");
	elemData = headerContainer.data();
	
	let snippetDOM = null;
	$.get(headerURL, function(htmlCode) {
		snippetDOM = $($.parseHTML(htmlCode));
		snippetDOM.find("div.title-insert").html(elemData['title']);
		headerContainer.html(snippetDOM.prop("outerHTML"));
	});
}

function insertFooter(footerURL, bodyData) {
	if (!bodyData.hasOwnProperty("depth")) {
		throw `Document-depth not speficied.  [${location.pathname.split("/").pop()}]`;
	}
	
	let snippetDOM = null;
	$.get(footerURL, function(htmlCode) {
		snippetDOM = $($.parseHTML(htmlCode));
		updateURLs(snippetDOM, bodyData);
		$("body > .footer-insert").html(snippetDOM.prop("outerHTML"));
	});

}

function insertNavbar(navbarURL, bodyData) {
	if (!bodyData.hasOwnProperty("depth")) {
		throw `Document-depth not speficied.  [${location.pathname.split("/").pop()}]`;
	}
	
	let snippetDOM = null;
	$.get(navbarURL, function(htmlCode){
		snippetDOM = $($.parseHTML(htmlCode));
		updateURLs(snippetDOM, bodyData);
		$("body > .navbar-insert").html(snippetDOM.prop("outerHTML"));
	});

}

$(document).ready(function() {
	let bodyData = $("body").data();
	try {
		insertNavbar("../html/navbar.html", bodyData);
		insertHeader("../html/header.html");
		insertFooter("../html/footer.html", bodyData);
	} catch (error) {
		$(".error-insert").html(insertError(error));
	}
});
