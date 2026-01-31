/**
 * Config.gs
 * Project constants and property retrieval.
 */

const props = PropertiesService.getScriptProperties();
const CLIENT_ID = props.getProperty('CLIENT_ID');
const CLIENT_SECRET = props.getProperty('CLIENT_SECRET');
const CLUB_ID = props.getProperty('CLUB_ID');
const DISCORD_WEBHOOK_URL = props.getProperty('DISCORD_WEBHOOK_URL');

const DISCORD_IDS = {
  'Srikar': '258466487012950018',
  'Wilco': '295065586797510658',
  'Trisan': '481967966657839107',
  'Jared': '273345887550439425',
  'Raymond': '581670531284074496',
  'Minnie': '608938431720194069',
  'Grace': '975644415852957736',
  'Chaomin': '608938431720194069',
  'Ravi': '328750094016970752',
  'Andy': '529931472233168908',
  'Scott': '635006091734024192',
  'Ben': '326226967160553472',
  'Tommy': '251433215481348096'
};

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
