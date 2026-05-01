const https = require("https");
const fs = require("fs");

https.get("https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/gpl-3.0.txt", (res) => {
    let data = "";
    res.on("data", (chunk) => { data += chunk; });
    res.on("end", () => {
        fs.writeFileSync("LICENSE", data);
        console.log("LICENSE written successfully");
    });
});
