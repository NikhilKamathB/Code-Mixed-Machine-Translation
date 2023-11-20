import re


def remove_urls(text: str) -> str:
    """Remove URLs from a text string"""
    return re.sub(r'http\S+', '', text)

def remove_usernames(text: str) -> str:
    """Remove usernames placeholders from a text string"""
    return text.replace('<user>', '').strip()

def remove_rt(text: str) -> str:
    """Remove retweet indications from a text string"""
    return text.replace('RT :', '').strip()

def remove_hashtags_and_mentions(text: str, retain_text: bool = False) -> str:
    """Remove hashtags and mentions. If retain_text=True, retains the text without # and @ symbols"""
    if retain_text:
        text = re.sub(r'#', '', text)
        return text
    else:
        text = re.sub(r'#\S+', '', text)
        return re.sub(r'@\S+', '', text)

def handle_emojis(text: str, remove: bool = False) -> str:
    """Handle emojis. If remove=True, removes all emojis."""
    if remove:
        # This pattern matches most common emojis. Depending on the specific dataset, it might need adjustments.
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"  # flags (iOS)
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    else:
        # You can expand this to convert specific emojis to text if needed
        return text
    
def remove_non_alpha(text: str) -> str:
    """Remove all non-alphabetical characters  from a text string."""
    # Use regular expression to replace all non-alphabet characters with an empty string
    return re.sub('[^a-zA-Z\s]', '', text)

def standardize_punctuation(text: str) -> str:
    """Standardize punctuation marks within the text"""
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\.+', '.', text)
    return re.sub(r'\?+', '?', text)

def to_lowercase(text: str) -> str:
    """Convert text to lowercase"""
    return text.lower()

def trim_extra_spaces(text: str) -> str:
    """Trim leading, trailing, and extra spaces within the text"""
    return ' '.join(text.split())

def clean_text(text: str, retain_hashtag_text: bool = False, remove_emoji: bool = False) -> str:
    """Aggregate function to clean the provided text based on individual cleaning functions"""
    text = remove_urls(text)
    text = remove_usernames(text)
    text = remove_rt(text)
    text = remove_hashtags_and_mentions(text, retain_hashtag_text)
    text = handle_emojis(text, remove_emoji)
    text = remove_non_alpha(text)
    text = standardize_punctuation(text)
    text = to_lowercase(text)
    text = trim_extra_spaces(text)
    return text