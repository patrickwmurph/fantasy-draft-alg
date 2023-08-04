import pandas as pd

# Load data
df = pd.read_csv('../2023-draft-order.csv')

# Clean the data
df_clean = df.dropna()

# Define the number of teams and rounds
num_teams = 12
num_rounds = 14

# Set our team's position in the draft order (0-indexed)
our_team_num = 0  # Change this variable to simulate the draft from different positions

# Define the required positions and their counts
required_positions = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

# Initialize a dict to hold the positions filled for our team
our_positions_filled = {pos: 0 for pos in required_positions}

# Define a function to pick a player for our team
def pick_our_player(available_players):
    # Loop over the players in order of VORP rank
    for _, player in available_players.iterrows():
        # If we have not yet filled all spots for this player's position, pick this player
        if our_positions_filled[player["Position"]] < required_positions[player["Position"]]:
            # Update our_positions_filled
            our_positions_filled[player["Position"]] += 1
            
            # Return the player's name
            return player["Player"]
    
    # If we have filled all spots for every position, just pick the top-ranked player
    return available_players.iloc[0]["Player"]

# Define a function to pick a player for another team
def pick_other_player(available_players):
    # Just pick the player with the lowest ADP
    return available_players.loc[available_players["ADP"].idxmin()]["Player"]

# Initialize an empty DataFrame to hold the available players
available_players = df_clean.copy()

# Reset our_positions_filled for our team
our_positions_filled = {pos: 0 for pos in required_positions}

# Clear the draft_picks
draft_picks = []

# Define a list to hold the order of the teams for the first round (1-12)
team_order = list(range(num_teams))  # 0 to 11 corresponding to 1 to 12

# Re-run the draft simulation
for round_num in range(num_rounds):
    round_picks = []  # Initialize an empty list to hold the picks for this round
    
    # Loop over the teams in the current order
    for team_num in team_order:
        # If it is our turn to pick (we are our_team_num), use our strategy
        if team_num == our_team_num:
            pick_func = pick_our_player
        # Otherwise, use the other strategy
        else:
            pick_func = pick_other_player
        
        # Pick a player
        player = pick_func(available_players)
        
        # Add the pick to round_picks
        round_picks.append({"Round": round_num + 1, "Team": team_num + 1, "Player": player})
        
        # Remove the player from available_players
        available_players = available_players[available_players["Player"] != player]
    
    # Add the picks for this round to draft_picks
    draft_picks.extend(round_picks)

    # Reverse the order of the teams for the next round (to simulate a snake draft)
    team_order.reverse()

# Convert draft_picks to a DataFrame
draft_picks_df = pd.DataFrame(draft_picks)

draft_picks_df = pd.merge(draft_picks_df, df[['Player', 'Position', 'PPR_Projected_2023']], on = 'Player', how = 'inner')

team_totals_df = draft_picks_df.groupby('Team').agg({'PPR_Projected_2023' : ['mean','sum']})

team_totals_df.sort_values(by = ('PPR_Projected_2023', 'sum'), ascending = False)

