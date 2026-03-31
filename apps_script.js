// Google Apps Script — paste this into script.google.com
// Then: Deploy → New deployment → Web app → Execute as "Me" → Access "Anyone"
//
// After deploying, set your GitHub token:
//   1. In Apps Script editor: Project Settings (gear icon) → Script Properties
//   2. Add property: GITHUB_TOKEN = your fine-grained PAT

function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    var email = (data.email || "").trim().toLowerCase();

    if (!email || email.indexOf("@") === -1) {
      return response_(400, "Invalid email");
    }

    var token = PropertiesService.getScriptProperties().getProperty("GITHUB_TOKEN");
    if (!token) {
      return response_(500, "Server misconfigured");
    }

    // Trigger the add_subscriber.yml workflow via GitHub API
    var url = "https://api.github.com/repos/wyw2010/macro-regime-agent/actions/workflows/add_subscriber.yml/dispatches";
    var payload = {
      ref: "master",
      inputs: { email: email }
    };

    var options = {
      method: "post",
      contentType: "application/json",
      headers: { Authorization: "Bearer " + token },
      payload: JSON.stringify(payload),
      muteHttpExceptions: true
    };

    var resp = UrlFetchApp.fetch(url, options);
    var code = resp.getResponseCode();

    if (code === 204) {
      return response_(200, "Subscribed");
    } else {
      Logger.log("GitHub API error: " + code + " " + resp.getContentText());
      return response_(502, "Upstream error");
    }
  } catch (err) {
    Logger.log("Error: " + err);
    return response_(500, "Server error");
  }
}

function response_(statusCode, message) {
  return ContentService
    .createTextOutput(JSON.stringify({ status: statusCode, message: message }))
    .setMimeType(ContentService.MimeType.JSON);
}
