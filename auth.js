/**
 * Auth.gs
 * Strava OAuth2 authentication and callbacks.
 */

function getStravaService() {
  return OAuth2.createService('Strava')
    .setAuthorizationBaseUrl('https://www.strava.com/oauth/authorize')
    .setTokenUrl('https://www.strava.com/oauth/token')
    .setClientId(CLIENT_ID)
    .setClientSecret(CLIENT_SECRET)
    .setCallbackFunction('authCallback')
    .setPropertyStore(PropertiesService.getUserProperties())
    .setScope('read,activity:read_all');
}

function authCallback(request) {
  const service = getStravaService();
  const authorized = service.handleCallback(request);
  return HtmlService.createHtmlOutput(authorized ? 'Success! You can close this tab.' : 'Access Denied.');
}

function logAuthUrl() {
  const service = getStravaService();
  if (!service.hasAccess()) {
    Logger.log('Authorize here: %s', service.getAuthorizationUrl());
  } else {
    Logger.log('Authenticated.');
  }
}
