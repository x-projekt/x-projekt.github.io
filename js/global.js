function parseDate(dateString) {
    let parts = dateString.split("-");
    return new Date(parts[0], parts[1] - 1, parts[2]);
}

function printDate(dateObject) {
    return `${dateObject.toLocaleDateString(
        "en-US", {month: "long"})}, ${dateObject.toLocaleDateString("en-US", {year: "numeric"})}`;
}

function errorFragment() {
    return`		
		<div>
			<span class="error d-inline-block">
				Oops... something went wrong!
			</span>
			<div class="lead">
				There was an error while loading the page. We suggest you do the following to
				diagnose the issue:
				<ol>
					<li>Refresh the page.</li>
					<li>(<small class="text-muted">if step-1 doesn't work</small>) Try opening this
						link in incognito/private-browsing window. You may copy and paste the URL 
						from	the address bar.
						<ol type="a">
							<li>(<small class="text-muted">if step-2 works</small>) Clear your 
							browser Cache and reload the page in a non-incognito window.</li>
						</ol>
					</li>
					<li>(<small class="text-muted">if step-3 doesn't work</small>) The problem
						could very well	be on our end. And we apologize for any and all
						inconvenience caused, while we work to solve it.</li>
				</ol>
			</div>
		</div>`;
}

function insertError(error) {
	let snippetDOM = null;
	$.get("../html/error.html", function(htmlCode) {
		snippetDOM = $($.parseHTML(htmlCode));
	});
	snippetDOM.find(".error-message-insert").append(error);
	
	$(".error-insert").html(snippetDOM.prop("outerHTML"));
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
		throw `Non-integer document depth ${depth}. [${location.pathname.split("/").pop()}]`
	}

	return depthPrefix.repeat(parseInt(depth)).concat(rootPages[urlId]);
}

function updateURLs(dom, bodyData) {
	let href = null, url = null;
	return dom.find("a").each(function () {
		href = $(this).attr("href");
		if (href.indexOf(affix) === 0 && href.lastIndexOf(affix) === href.length - 1) {
			url = getURL(href.substr(1, href.length - 2), bodyData["depth"]);
			$(this).attr("href", url);
		}
	});
}

function insertHeader(headerURL) {
	let headerContainer = $("body > .header-insert");
	elemData = headerContainer.data();
	
	let snippetDOM = null;
	$.get(headerURL, function(htmlCode) {
		snippetDOM = $($.parseHTML(htmlCode));
	});
	snippetDOM.find("div.title-insert").html(elemData['title']);

	headerContainer.html(snippetDOM.prop("outerHTML"));
}

function insertFooter(footerURL, bodyData) {
	if (!bodyData.hasOwnProperty("depth")) {
		throw `Depth attribute not speficied.  [${location.pathname.split("/").pop()}]`;
	}
	
	let snippetDOM = null;
	$.get(footerURL, function(htmlCode) {
		snippetDOM = $($.parseHTML(htmlCode));
	});
	snippetDOM = updateURLs(snippetDOM, bodyData);

	$("body > .footer-insert").html(snippetDOM.prop("outerHTML"));
}

function insertNavbar(navbarURL, bodyData) {
	if (!bodyData.hasOwnProperty("depth")) {
		throw `Depth attribute not speficied.  [${location.pathname.split("/").pop()}]`;
	}
	
	let snippetDOM = null;
	$.get(navbarURL, function(htmlCode){
		snippetDOM = $($.parseHTML(htmlCode));
	});
	snippetDOM = updateURLs(snippetDOM, bodyData);

	$("body > .navbar-insert").html(snippetDOM.prop("outerHTML"));
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
