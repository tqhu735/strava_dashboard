/**
 * Config.gs
 * Project constants and property retrieval.
 */

const props = PropertiesService.getScriptProperties();
const CLIENT_ID = props.getProperty('CLIENT_ID');
const CLIENT_SECRET = props.getProperty('CLIENT_SECRET');
const CLUB_ID = props.getProperty('CLUB_ID');
const DISCORD_WEBHOOK_URL = props.getProperty('DISCORD_WEBHOOK_URL');

const TEAMS = {
  'Srikar': ['Srikar', 'Wilco', 'Trisan', 'Jared', 'Raymond', "Minnie", "Grace", "Chaomin"],
  'Ravi': ['Ravi', 'Andy', 'Scott', 'Ben', 'Tommy']
};

const MULTIPLIERS = {
  'Hike': 0.5,
  'Ride': 0.3,
  'Rowing': 0.75,
  'Run': 1.0,
  'Swim': 4.0,
  'Walk': 0.2,
  'Workout': 0.3
};

const DISCORD_COLOR = 15548997; // Strava-esque Orange/Red
