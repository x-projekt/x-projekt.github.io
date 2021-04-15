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
