import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import yaml

#read the config file and get input/output paths
with open('paths.yml', 'r') as file:
    config = yaml.safe_load(file)

input_path = config['your_paths']['input_path']
output_path = config['your_paths']['output_path']

#read in excel spreadsheet
sheets = pd.read_excel(input_path, sheet_name=None, header=None)

#define the values we want in column E (index 4)
valid_values = ['conM', 'conMNM', 'conNM', 'incM', 'incMNM', 'incNM']

filtered_sheets = {}

#loop over each sheet and filter data based on valid values in column E
for sheet_name, sheet_data in sheets.items():
    #filter the sheet based on the values in column E (index 4)
    filtered_data = sheet_data[sheet_data.iloc[:, 4].isin(valid_values)]
    filtered_data = filtered_data.iloc[:, :8]
    filtered_data.columns = ['Trial', 'RunLabel', 'Condition', 'TrialStart', 'EventTag', 'Time', 'keys', 'match_status']
    filtered_sheets[sheet_name] = filtered_data

for sheet_name, sheet_data in filtered_sheets.items():
    #drop duplicates to get rid of extra colnames
    sheet_data = sheet_data.drop_duplicates()
    #update the filtered_sheets with the cleaned data
    filtered_sheets[sheet_name] = sheet_data

#define the values we want in column E
valid_values = ['conM', 'conMNM', 'conNM', 'incM', 'incMNM', 'incNM', 'EventTag']

#define the colnames
valid_row_values = ['Trial', 'RunLabel', 'Condition', 'trial start', 'EventTag', 'Time', 'keys', 'sequence', 'mouse_down']

filtered_sheets_2 = {}

#filter each sheet based on valid values in column E (index 4)
for sheet_name, sheet_data in sheets.items():
    filtered_data = sheet_data[sheet_data.iloc[:, 4].isin(valid_values)]
    filtered_data = filtered_data.reset_index(drop=True)
    filtered_sheets_2[sheet_name] = filtered_data

#set colnames, subset first 15 rows and delete dupes
for sheet_name, sheet_data in filtered_sheets_2.items():

    #check to see if the sheet is empty after previous filtering
    if len(sheet_data) == 0:
        print(f"Warning: Sheet '{sheet_name}' is empty. Skipping.")
        continue  #skip to next sheet
    
    sheet_data.columns = sheet_data.iloc[0]
    sheet_data = sheet_data.drop(index=0).reset_index(drop=True)
    
    first_15_rows = sheet_data.head(15)
    first_15_rows = first_15_rows.drop_duplicates(keep='first')

    sheet_data = pd.concat([first_15_rows, sheet_data.iloc[15:]]).reset_index(drop=True)
    filtered_sheets_2[sheet_name] = sheet_data

#function that splits the data into separate blocks (individual dfs) for each run (based on each occurrence of header row)
def split_dataframe_by_header(sheet_data):
    blocks = []
    header_indices = sheet_data[sheet_data.iloc[:, 0] == valid_row_values[0]].index.tolist()
    header_indices.append(len(sheet_data))

    #split the df into blocks
    for i in range(len(header_indices) - 1):
        start = header_indices[i]
        end = header_indices[i + 1]
        block = sheet_data.iloc[start:end].reset_index(drop=True)
        
        #if first row of the block is a header, drop it (already used as colname)
        if block.iloc[0].tolist() == valid_row_values:
            block = block.drop(index=0).reset_index(drop=True)

        blocks.append(block)

    return blocks

#test_subj = filtered_sheets_2['HP23-01696']

#function to remove any extra characters in 'keys' column (such as _5UP)
#def clean_keys_column(sheet_data):
    #remove rows where there are extra characters (other than 5_up) because they are invalid. 
    
    #keeps only the text between the first [] for remaining, which should be only _5UP 
#    sheet_data['keys'] = sheet_data['keys'].str.replace(r'\[([0-9]+)\].*', r'[\1]', regex=True)
#    return sheet_data

def clean_keys_column(sheet_data):
    
    #debug prints
    print("Columns in sheet_data:", sheet_data.columns.tolist())  
    print("First few rows:\n", sheet_data.head())  
    """
    Cleans 'keys' column by:
    - Extracting the first [int] if the row matches:
        - Single [int] (e.g., [1]), OR
        - Multiple [int]s ending with _UP (e.g., [1][5]_UP).
    - Leaves other rows unchanged (e.g., "abc", "[1]xyz").
    """
    # Pattern to match valid cases (single [int] or [int]..._UP)
    valid_pattern = r'^(\[\d+\])(?:$|(\[\d+\])*_UP$)'
    
    # Extract first [int] ONLY for valid rows
    sheet_data['keys'] = sheet_data['keys'].str.extract(r'^(\[\d+\])', expand=False).fillna("")
    
    return sheet_data

#function to find the most common value in 'keys' for specific EventTag section
def most_common_key_in_section(block, event_tag):
    section_keys = block[block['EventTag'] == event_tag]['keys']
    mode_result = section_keys.mode()
    return mode_result[0] if not mode_result.empty else None



def map_condition(condition_value, match_key='[1]', nonmatch_key='[2]'):
    """
    Maps conditions to expected responses based on participant's response pattern.
    """
    if pd.isna(condition_value):
        return condition_value
    
    if 'conM' in condition_value or 'incM' in condition_value:
        return match_key
    elif 'conNM' in condition_value or 'incNM' in condition_value:
        return nonmatch_key
    else:
        return condition_value
    

#function to replace most common keys in the block
def replace_most_common_keys(block, conM_key, conNM_key):
    if conM_key:
        block.loc[:, 'keys'] = block['keys'].replace(conM_key, '[1]')
    if conNM_key:
        block.loc[:, 'keys'] = block['keys'].replace(conNM_key, '[2]')
    return block

#function to drop extra columns if more than 9
def drop_extra_columns(sheet_data):
    if sheet_data.shape[1] > 9:
        sheet_data = sheet_data.drop(columns=sheet_data.columns[9])
        #print(f"Extra column dropped.")
    return sheet_data

def process_block_data(block):
    block = clean_block_data(block)
    
    #convert time to numeric
    block['Time'] = pd.to_numeric(block['Time'], errors='coerce')
    
    #remove missing values
    block = block[block['keys'].notna()]
    
    #determine participant's response mapping
    match_key, nonmatch_key = determine_response_mapping(block)
    
    if not match_key or not nonmatch_key:
        match_key = '[1]'
        nonmatch_key = '[2]'
    
    block = block.copy()
    #use the determined mapping to set conditions
    block['Condition'] = block['Condition'].apply(
        lambda x: map_condition(x, match_key=match_key, nonmatch_key=nonmatch_key)
    )
    
    #count errors with correct mapping
    error_mask = block['Condition'] != block['keys']
    error_count = error_mask.sum()
    
    #subset errors
    error_block = block[error_mask]
    
    #remove errors
    block = block[~error_mask]
    
    return block, error_count, error_block

#function to clean the block data (drop last two columns, drop rows that match header row, etc.)
def clean_block_data(block):
    block = block.iloc[:, :-2]  
    if block.iloc[-1].tolist() == valid_row_values:
        block = block.drop(block.index[-1])  
    return block

#function to remove any extra characters in 'keys' column (such as _5UP) in block
def clean_keys_column_block(block):
    block['keys'] = block['keys'].str.replace(r'\[([0-9]+)\].*', r'[\1]', regex=True)
    return block

def update_subject_metrics(block, subject_metrics):
    #get error count and cleaned block
    block_cleaned, errors_in_block, error_block = process_block_data(block)
    subject_metrics['total_errors'] += errors_in_block

    #use cleaned block for remaining calculations
    real_missed_trials, fake_missed_trials, duplicated_trials = count_trials(block_cleaned)
    subject_metrics['real_missed_trials'] += real_missed_trials
    subject_metrics['fake_missed_trials'] += fake_missed_trials
    subject_metrics['duplicated_trials'] += duplicated_trials

    subject_metrics['total_time'] += block_cleaned['Time'].sum()
    subject_metrics['total_rows'] += len(block_cleaned)
    
    return block_cleaned

def determine_response_mapping(block):
    #get responses for match and nonmatch conditions
    match_responses = block[block['EventTag'].str.contains('M', na=False) & 
                          ~block['EventTag'].str.contains('NM', na=False)]['keys']
    nonmatch_responses = block[block['EventTag'].str.contains('NM', na=False)]['keys']
    
    #find most common response for each condition
    match_key = match_responses.mode().iloc[0] if not match_responses.empty else None
    nonmatch_key = nonmatch_responses.mode().iloc[0] if not nonmatch_responses.empty else None
    
    return match_key, nonmatch_key

#function to count real missed trials, fake missed trials, and duplicated trials
def count_trials(block):
    trial_numbers = block['Trial'].astype(int)
    event_tags = block['EventTag']

    real_missed_trials = 0
    fake_missed_trials = 0
    duplicated_trials = 0

    for i in range(1, len(trial_numbers)):
        prev_trial = trial_numbers.iloc[i - 1]
        curr_trial = trial_numbers.iloc[i]
        prev_event = event_tags.iloc[i - 1]
        curr_event = event_tags.iloc[i]

        if curr_trial > prev_trial + 1 and prev_event == curr_event:
            real_missed_trials += 1
        elif curr_trial > prev_trial + 1 and prev_event != curr_event:
            fake_missed_trials += 1

    duplicated_trials = trial_numbers[trial_numbers.duplicated()].nunique()
    return real_missed_trials, fake_missed_trials, duplicated_trials


#function to calculate the average time for the subject
def calculate_average_time(total_time, total_rows):
    return total_time / total_rows if total_rows else 0

#function to initialize the subject's metrics
def initialize_subject_metrics():
    return {
        'real_missed_trials': 0,
        'fake_missed_trials': 0,
        'duplicated_trials': 0,
        'total_errors': 0,
        'total_time': 0,
        'total_rows': 0,
        'average_time': 0
    }

def initialize_subject_metrics_2():
    return{
        'std': 0,
        '4std': 0,
        'outlier_cutoff': 0
    }
     
def process_subject_data(filtered_sheets_2):
    subject_trial_counts = {}
    processed_sheets = {}
    cleaned_sheets = {}
    error_sheets = {}

    for sheet_name, sheet_data in filtered_sheets_2.items():
        subject_metrics = initialize_subject_metrics()
        sheet_data = drop_extra_columns(sheet_data)
        sheet_data = clean_keys_column(sheet_data)
        blocks = split_dataframe_by_header(sheet_data)
        
        processed_blocks = []
        error_blocks = []
        
        for block in blocks:
            if not block.empty:
                block = clean_keys_column_block(block)
                #process the block and update metrics
                cleaned_block, errors_in_block, error_block = process_block_data(block)
                
                #update metrics
                subject_metrics['total_errors'] += errors_in_block
                
                #count trials for this block
                real_missed, fake_missed, duplicated = count_trials(cleaned_block)
                subject_metrics['real_missed_trials'] += real_missed
                subject_metrics['fake_missed_trials'] += fake_missed
                subject_metrics['duplicated_trials'] += duplicated
                
                #update time and row counts
                subject_metrics['total_time'] += cleaned_block['Time'].sum()
                subject_metrics['total_rows'] += len(cleaned_block)
                
                processed_blocks.append(cleaned_block)
                if not error_block.empty:
                    error_blocks.append(error_block)

        #combine all cleaned blocks
        if processed_blocks:
            processed_sheet = pd.concat(processed_blocks, ignore_index=True)
            processed_sheets[sheet_name] = processed_sheet
            
        #combine all error blocks for this subject
        if error_blocks:
            error_sheet = pd.concat(error_blocks, ignore_index=True)
            error_sheets[sheet_name] = error_sheet

        subject_metrics['average_time'] = calculate_average_time(
            subject_metrics['total_time'], 
            subject_metrics['total_rows']
        )
        
        #store the metrics for this subject
        subject_trial_counts[sheet_name] = subject_metrics

        #remove header rows
        if processed_sheets.get(sheet_name) is not None:
            header_mask = processed_sheets[sheet_name]['Trial'] == 'Trial'
            sheet_data = processed_sheets[sheet_name][~header_mask].copy()

            subject_metrics['std'] = sheet_data['Time'].std()
            subject_metrics['4std'] = 4 * subject_metrics['std']
            subject_metrics['outlier_cutoff'] = subject_metrics['4std'] + subject_metrics['average_time']

            #remove outliers
            outlier_mask = processed_sheets[sheet_name]['Time'] >= subject_metrics['outlier_cutoff']
            subject_metrics['total_outliers'] = outlier_mask.sum()
            cleaned_sheet = processed_sheets[sheet_name][~outlier_mask].copy()
            cleaned_sheets[sheet_name] = cleaned_sheet

            subject_metrics['mean'] = cleaned_sheet['Time'].mean()
            subject_metrics['median'] = cleaned_sheet['Time'].median()
            subject_metrics['stdev'] = cleaned_sheet['Time'].std()

    return subject_trial_counts, filtered_sheets_2, cleaned_sheets, error_sheets
                  
#call the main function
subject_trial_counts, filtered_sheets_2, cleaned_sheets, error_sheets = process_subject_data(filtered_sheets_2)

def analyze_errors(error_sheets):
    for subject, error_data in error_sheets.items():
        print(f"\nErrors for subject {subject}:")
        print(f"Total error trials: {len(error_data)}")
        
        print("\nError distribution by condition:")
        print(error_data['EventTag'].value_counts())
        
        print("\nSample of error trials:")
        print(error_data[['EventTag', 'Condition', 'keys', 'Time']].head())
        
        print("\n" + "="*50)

subject_trial_counts, filtered_sheets_2, cleaned_sheets, error_sheets = process_subject_data(filtered_sheets_2)

conditional_df = {}
conditional_stats = {}

for sheet_name, sheet_data in cleaned_sheets.items(): 
    conditional_df[sheet_name] = sheet_data.sort_values(by = 'EventTag', ascending = True)
    current_sorted = conditional_df[sheet_name]

    #conM-RR 
    conM_df = current_sorted[current_sorted['EventTag'] == 'conM']

    #incM-RR
    incM_df = current_sorted[current_sorted['EventTag'] == 'incM']

    #conM-RS 
    #df that is all rows other than the conM rr ones
    non_conM_df = current_sorted[current_sorted['EventTag'] != 'conM']
    #df that is all conM Conditions that are in non_conM_df aka all the rs conM's
    con_rs_df = non_conM_df[non_conM_df['EventTag'].str.contains('conM', case=False, na=False)]
    conM_rs_df = con_rs_df[con_rs_df['Condition'] == '[1]']

    #incM-RS
    rs_df = current_sorted[current_sorted['EventTag'].str.contains('MNM', case=False, na=False)]
    inc_rs_df = rs_df[rs_df['EventTag'].str.contains('inc', case=False, na=False)]
    incM_rs_df = inc_rs_df[inc_rs_df['Condition'] == '[1]']

    #conNM-RR
    conNM_df = current_sorted[current_sorted['EventTag'] == 'conNM']

    #incNM-RR
    incNM_df = current_sorted[current_sorted['EventTag'] == 'incNM']

    #conNM-RS
    #df that is all rows other than the conNM rr ones
    non_conNM_df = current_sorted[current_sorted['EventTag'] != 'conNM']
    #df that is all conNM Conditions that are in non_conNM_df aka all the rs conNM's
    #conNM_rs_df = non_conNM_df[non_conNM_df['EventTag'].str.contains('conNM', case=False, na=False)]
    conNM_rs_df = non_conNM_df[non_conNM_df['EventTag'] == 'conMNM']
    conNM_rs_df = conNM_rs_df[conNM_rs_df['Condition'] == '[2]']

    #incNM-RS
    #df that is all rows other than the incNM rr ones
    non_incNM_df = current_sorted[current_sorted['EventTag'] != 'incNM']
    #df that is all incNM Conditions that are in non_incNM_df aka all the rs incNM's
    incNM_rs_df = non_incNM_df[non_incNM_df['EventTag'] == 'incMNM']
    incNM_rs_df = incNM_rs_df[incNM_rs_df['Condition'] == '[2]']

    #con-RR
    #remove response switching
    rr_df = current_sorted[~current_sorted['EventTag'].str.contains('MNM', case=False, na=False)]
    con_rr_df = rr_df[rr_df['EventTag'].str.contains('con', case=False, na=False)]

    #inc-RR
    inc_rr_df = rr_df[rr_df['EventTag'].str.contains('inc', case=False, na=False)]

    #con-RS 
    #df that is all rows other than the conM rr ones
    non_conM_df = current_sorted[current_sorted['EventTag'] != 'conM']
    #df that is all conM Conditions that are in non_conM_df aka all the rs conM's
    con_rs_df = non_conM_df[non_conM_df['EventTag'].str.contains('conM', case=False, na=False)]

    #inc-RS
    #subset response switching
    rs_df = current_sorted[current_sorted['EventTag'].str.contains('MNM', case=False, na=False)]
    inc_rs_df = rs_df[rs_df['EventTag'].str.contains('inc', case=False, na=False)]

    #update conditional stats with this sheet's metrics
    conditional_stats[sheet_name] = {

        'conM-RR mean': conM_df['Time'].mean(),
        'conM-RR median': conM_df['Time'].median(),
        'conM-RR stdev': conM_df['Time'].std(),

        'incM-RR mean': incM_df['Time'].mean(),
        'incM-RR median': incM_df['Time'].median(),
        'incM-RR stdev': incM_df['Time'].std(),

        'conM-RS mean': conM_rs_df['Time'].mean(),
        'conM-RS median': conM_rs_df['Time'].median(),
        'conM-RS stdev': conM_rs_df['Time'].std(),

        'incM-RS mean': incM_rs_df['Time'].mean(),
        'incM-RS median': incM_rs_df['Time'].median(),
        'incM-RS stdev': incM_rs_df['Time'].std(),

        'conNM-RR mean': conNM_df['Time'].mean(),
        'conNM-RR median': conNM_df['Time'].median(),
        'conNM-RR stdev': conNM_df['Time'].std(),

        'incNM-RR mean': incNM_df['Time'].mean(),
        'incNM-RR median': incNM_df['Time'].median(),
        'incNM-RR stdev': incNM_df['Time'].std(),

        'conNM-RS mean': conNM_rs_df['Time'].mean(),
        'conNM-RS median': conNM_rs_df['Time'].median(),
        'conNM-RS stdev': conNM_rs_df['Time'].std(),

        'incNM-RS mean': incNM_rs_df['Time'].mean(),
        'incNM-RS median': incNM_rs_df['Time'].median(),
        'incNM-RS stdev': incNM_rs_df['Time'].std(),

        'con-RR mean': con_rr_df['Time'].mean(),
        'con-RR median': con_rr_df['Time'].median(),
        'con-RR stdev': con_rr_df['Time'].std(),

        'inc-RR mean': inc_rr_df['Time'].mean(),
        'inc-RR median': inc_rr_df['Time'].median(),
        'inc-RR stdev': inc_rr_df['Time'].std(),

        'con-RS mean': con_rs_df['Time'].mean(),
        'con-RS median': con_rs_df['Time'].median(),
        'con-RS stdev': con_rs_df['Time'].std(),

        'inc-RS mean': inc_rs_df['Time'].mean(),
        'inc-RS median': inc_rs_df['Time'].median(),
        'inc-RS stdev': inc_rs_df['Time'].std(),
    }

'''
print tallied results for each subject
for subject, counts in subject_trial_counts.items():
    print(f"\nTally for {subject}:")
    print(f"  - Real missed trials: {counts['real_missed_trials']}")
    print(f"  - Fake missed trials (block switches): {counts['fake_missed_trials']}")
    print(f"  - Duplicate trials: {counts['duplicated_trials']}")
    print(f"  - Errors: {counts['total_errors']}")
    print(f"  - Mean Time Errors Removed: {counts['average_time']:.3f}")
    print(f"  - Standard Deviation Errors Removed: {counts['std']:.3f}")
    print(f"  - Outlier Cutoff: {counts['outlier_cutoff']:.3f}")
    print(f"  - Number outliers (prolonged): {counts['total_outliers']}")
    print(f"  - Mean Time Outliers and Errors Removed: {counts['mean']:.2f}")
    print(f"  - Median Outliers and Errors Removed: {counts['median']:.2f}")
    print(f"  - SD Time Outliers and Errors Removed: {counts['stdev']:.2f}")

#export conditional stats to csv with subject_id
conditional_stats = pd.DataFrame.from_dict(conditional_stats).T
conditional_stats.to_csv('/Users/kjung6/Eva/Stroop/final_dataset/2-11-25_conditional_stats.csv', index_label='Subject_ID')

#export subject trial counts as csv
subject_trial_counts = pd.DataFrame.from_dict(subject_trial_counts)
subject_trial_counts.to_csv('/Users/kjung6/Eva/Stroop/final_dataset/2-11-25_subject_trial_counts.csv', index_label='Statistics')

#export subject trial counts as csv TRANSPOSED
subject_trial_counts = pd.DataFrame.from_dict(subject_trial_counts).T
subject_trial_counts.to_csv('/Users/kjung6/Eva/Stroop/final_dataset/2-11-25_subject_trial_counts_T.csv', index_label='Subject_ID')
'''

#convert to pd df, make Subject_ID the first column, then combine
subject_trial_counts = pd.DataFrame.from_dict(subject_trial_counts).T
subject_trial_counts = subject_trial_counts.reset_index()
subject_trial_counts = subject_trial_counts.rename(columns = {'index' : 'Subject_ID'})

conditional_stats = pd.DataFrame.from_dict(conditional_stats).T
conditional_stats = conditional_stats.reset_index()
conditional_stats = conditional_stats.rename(columns = {'index' : 'Subject_ID'})
combined_stats = pd.merge(subject_trial_counts, conditional_stats, on = "Subject_ID", how = "left")

#export final combined stats
combined_stats.to_csv(f'{output_path}')
