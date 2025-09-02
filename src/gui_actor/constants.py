# FiftyOne-specific constants
IGNORE_INDEX = -100
DEFAULT_POINTER_START_TOKEN = "<|pointer_start|>"
DEFAULT_POINTER_END_TOKEN = "<|pointer_end|>"
DEFAULT_POINTER_PAD_TOKEN = "<|pointer_pad|>"

# Simple, clear system message
grounding_system_message = "You are a GUI agent for FiftyOne. Respond with actions and coordinates in JSON format."

# Remove coordinate patterns from text - let JSON handle it
ACTION_PATTENS_JSON = [
    r'"x":\s*([\d.]+)',
    r'"y":\s*([\d.]+)',
]

ADDITIONAL_SPECIAL_TOKENS = [
    "<|recipient|>",
    "<|diff_marker|>",
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
]

# Consistent ending token
RESPONSE_END_TOKEN = "<|diff_marker|>"

# Simplified template
chat_template = """{% for message in messages %}<|im_start|>{{ message['role'] }}{% if 'recipient' in message %}<|recipient|>{{ message['recipient'] }}{% endif %}
{{ message['content'][0]['text'] }}<|diff_marker|>
{% endfor %}"""