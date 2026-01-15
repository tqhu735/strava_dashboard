/**
 * Main.gs
 * Core logic for fetching activities and sending notifications.
 */

/**
 * Main routine: Fetches, filters, and logs new activities.
 */
function fetchClubActivities() {
    const service = getStravaService();
    if (!service.hasAccess()) {
        Logger.log('Authentication lost. Run logAuthUrl().');
        return;
    }

    const url = `https://www.strava.com/api/v3/clubs/${CLUB_ID}/activities?per_page=200`;
    let activities;

    try {
        const response = UrlFetchApp.fetch(url, {
            headers: { Authorization: `Bearer ${service.getAccessToken()}` }
        });
        activities = JSON.parse(response.getContentText());
    } catch (e) {
        Logger.log(`Fetch error: ${e.message}`);
        return;
    }

    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Data');
    const lastRow = sheet.getLastRow();
    const existingIds = lastRow > 1
        ? sheet.getRange(2, 1, lastRow - 1, 1).getValues().flat().map(String)
        : [];

    const newRows = activities
        .map(act => {
            const athleteKey = `${act.athlete.firstname}_${act.athlete.lastname}`.replace(/\s+/g, '_');
            const uniqueId = `${athleteKey}_${act.distance}_${act.moving_time}_${act.name}`.replace(/\s+/g, '_');

            if (existingIds.includes(uniqueId)) return null;

            const dateStr = act.start_date_local || act.start_date;
            const date = dateStr ? new Date(dateStr) : new Date();
            const distKm = (act.distance || 0) / 1000;
            const effectiveDistKm = getEffectiveDistance(distKm, act.type);
            const durationMin = (act.moving_time || 0) / 60;
            const paceDecimal = distKm > 0 ? Number((durationMin / distKm).toFixed(2)) : 0;
            const firstName = toTitleCase(act.athlete.firstname);
            const team = getTeam(firstName);

            return [
                uniqueId, firstName, team, date,
                Number(distKm.toFixed(2)), effectiveDistKm,
                Number(durationMin.toFixed(2)), paceDecimal,
                act.total_elevation_gain || 0, act.type
            ];
        })
        .filter(row => row !== null);

    if (newRows.length > 0) {
        sheet.getRange(lastRow + 1, 1, newRows.length, newRows[0].length).setValues(newRows);
        Logger.log(`Added ${newRows.length} items.`);
        sendDiscordNotification(newRows);
    } else {
        Logger.log('No new activities found.');
    }
}

/**
 * Sends a formatted notification to Discord for new activities.
 */
function sendDiscordNotification(newActivities) {
    if (!DISCORD_WEBHOOK_URL) {
        Logger.log('DISCORD_WEBHOOK_URL not set. Skipping notification.');
        return;
    }

    const filteredActivities = newActivities.filter(act => {
        const dist = act[4];
        const type = act[9];
        return type !== 'Walk' || dist > 1.0;
    });

    if (filteredActivities.length === 0) {
        Logger.log('No qualifying activities for Discord notification.');
        return;
    }

    const embeds = filteredActivities.map(act => {
        const [id, name, team, date, dist, effDist, duration, pace, elevation, type] = act;
        const multiplier = MULTIPLIERS[type] || 0.0;

        return {
            title: `${name} recorded a ${type}!`,
            color: DISCORD_COLOR,
            fields: [
                { name: 'Distance', value: `${dist.toFixed(2)} km`, inline: true },
                { name: 'Multiplier', value: `${multiplier.toFixed(2)}x`, inline: true },
                { name: 'Effort', value: `${effDist.toFixed(2)}`, inline: true },
                { name: 'Duration', value: `${duration.toFixed(2)} min`, inline: true },
                { name: 'Pace', value: `${formatDuration(pace)}/km`, inline: true },
                { name: 'Elevation', value: `${elevation} m`, inline: true },
                { name: 'Team', value: team, inline: true },
            ],
            timestamp: new Date(date).toISOString()
        };
    });

    for (let i = 0; i < embeds.length; i += 10) {
        const chunk = embeds.slice(i, i + 10);
        const payload = {
            content: embeds.length > 10 ? `Batch update: ${filteredActivities.length} new activities!` : null,
            embeds: chunk
        };

        const options = {
            method: 'post',
            contentType: 'application/json',
            payload: JSON.stringify(payload)
        };

        try {
            UrlFetchApp.fetch(DISCORD_WEBHOOK_URL, options);
            Logger.log(`Successfully sent ${chunk.length} activities to Discord.`);
        } catch (e) {
            Logger.log(`Discord notification error: ${e.message}`);
        }
    }
}
