import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from dateutil import parser as date_parser

load_dotenv()

# Calendly configuration from environment or defaults
DEFAULT_CALENDLY_URL = "https://calendly.com/neerajjawahirani/30min"
CALENDLY_URL = os.getenv("CALENDLY_URL", DEFAULT_CALENDLY_URL)
CALENDLY_API_TOKEN = os.getenv("CALENDLY_API_TOKEN", "")
TIMEZONE_OFFSET = os.getenv("CALENDLY_TIMEZONE_OFFSET", "-04:00")


class CalendlyBookingHandler:
    """
    Handles appointment booking flow via Calendly integration.
    - Asks for missing date or time details if not provided.
    - Generates pre-selected direct slot booking links and standard scheduling links once date & time are given.
    """

    def __init__(self, calendly_url: Optional[str] = None, api_token: Optional[str] = None):
        self.calendly_url = (calendly_url or CALENDLY_URL).rstrip("/")
        self.api_token = api_token or CALENDLY_API_TOKEN
        self.tz_offset = TIMEZONE_OFFSET

    def is_booking_request(self, query: str, history: Optional[list] = None) -> bool:
        """
        Determines whether the query or context relates to booking an appointment.
        """
        keywords = [
            "book", "appointment", "schedule", "meeting", "meet", "call", 
            "slot", "calendly", "reservation", "consultation", "session"
        ]
        query_lower = query.lower()
        
        # Direct keyword match
        if any(kw in query_lower for kw in keywords):
            return True
            
        # Check if previous assistant message was asking for date/time
        if history and len(history) > 0:
            last_turn = history[-1]
            last_assistant_msg = last_turn.get("assistant", "").lower()
            if "date" in last_assistant_msg or "time" in last_assistant_msg or "appointment" in last_assistant_msg or "schedule" in last_assistant_msg:
                # Check if current user input contains date/time responses
                if self._has_date_or_time_indicators(query_lower):
                    return True
                    
        return False

    def _has_date_or_time_indicators(self, text: str) -> bool:
        """
        Helper to detect date or time related words.
        """
        date_time_words = [
            "today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", 
            "friday", "saturday", "sunday", "am", "pm", "clock", "morning", 
            "afternoon", "evening", "night", "january", "february", "march", 
            "april", "may", "june", "july", "august", "september", "october", 
            "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", 
            "aug", "sep", "oct", "nov", "dec", "next week", "this week"
        ]
        if any(w in text for w in date_time_words):
            return True
        # Check for numbers like 3pm, 10:30, 2026-09-25, 25th
        if re.search(r'\d{1,2}(:\d{2})?\s*(am|pm)?', text) or re.search(r'\d{1,2}(st|nd|rd|th)', text):
            return True
        return False

    def extract_datetime_details(self, query: str, history: Optional[list] = None, llm_client: Any = None) -> Dict[str, Any]:
        """
        Extracts date and time information from query and context, resolving to normalized ISO format.
        Returns a dict:
        {
            "has_date": bool,
            "has_time": bool,
            "display_date": str,
            "display_time": str,
            "iso_date": str,  # YYYY-MM-DD
            "iso_time": str   # HH:MM:SS
        }
        """
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")
        current_day_str = now.strftime("%A")

        if llm_client:
            try:
                history_str = ""
                if history:
                    history_str = "\n".join([f"User: {h.get('user', '')}\nAssistant: {h.get('assistant', '')}" for h in history[-3:]])

                extraction_prompt = f"""
You are an entity extractor for an appointment booking assistant.
Current Reference Date: {current_date_str} ({current_day_str})
Current Timezone Offset: {self.tz_offset}

Analyze the conversation history and user query to check if a DATE and TIME for an appointment are mentioned.
If relative dates (e.g. "tomorrow", "next Wednesday", "Friday", "Sept 23") or times (e.g. "10:30 AM", "3 PM", "15:00") are mentioned, resolve them strictly relative to Current Reference Date: {current_date_str}.

Respond strictly with a JSON object in this format (no markdown, no extra text):
{{
    "has_date": true/false,
    "has_time": true/false,
    "display_date": "Human readable date (e.g. Wednesday, Sep 23, 2026) or empty string",
    "display_time": "Human readable time (e.g. 10:30 AM) or empty string",
    "iso_date": "YYYY-MM-DD format or empty string",
    "iso_time": "HH:MM:SS 24-hour format or empty string"
}}
"""
                response = llm_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": "You output JSON only."},
                        {"role": "user", "content": f"HISTORY:\n{history_str}\n\nUSER QUERY:\n{query}\n\n{extraction_prompt}"}
                    ],
                    temperature=0.0,
                    max_tokens=250
                )
                raw_text = response.choices[0].message.content.strip()
                raw_text = re.sub(r'```json\s*', '', raw_text)
                raw_text = re.sub(r'```\s*$', '', raw_text).strip()
                data = json.loads(raw_text)

                # Validate iso formats
                iso_d = data.get("iso_date", "").strip()
                iso_t = data.get("iso_time", "").strip()

                if iso_d and not re.match(r'^\d{4}-\d{2}-\d{2}$', iso_d):
                    iso_d = ""
                if iso_t and not re.match(r'^\d{2}:\d{2}:\d{2}$', iso_t):
                    if re.match(r'^\d{2}:\d{2}$', iso_t):
                        iso_t += ":00"
                    else:
                        iso_t = ""

                return {
                    "has_date": bool(data.get("has_date", False)) and bool(iso_d or data.get("display_date")),
                    "has_time": bool(data.get("has_time", False)) and bool(iso_t or data.get("display_time")),
                    "display_date": data.get("display_date", "").strip(),
                    "display_time": data.get("display_time", "").strip(),
                    "iso_date": iso_d,
                    "iso_time": iso_t
                }
            except Exception:
                pass

        # Rule-based fallback extraction
        return self._rule_based_extract(query, history, now)

    def _rule_based_extract(self, query: str, history: Optional[list] = None, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Rule-based fallback for date and time parsing.
        """
        now = now or datetime.now()
        full_text = query
        if history and len(history) > 0:
            past_turns = [h.get("user", "") for h in history[-2:]]
            full_text = f"{' '.join(past_turns)} {query}"

        text_lower = full_text.lower()

        has_date = False
        iso_date = ""
        display_date = ""

        # Relative date keywords
        if "tomorrow" in text_lower:
            target_dt = now + timedelta(days=1)
            has_date = True
            iso_date = target_dt.strftime("%Y-%m-%d")
            display_date = target_dt.strftime("%A, %b %d, %Y")
        elif "today" in text_lower:
            has_date = True
            iso_date = now.strftime("%Y-%m-%d")
            display_date = now.strftime("%A, %b %d, %Y")
        else:
            # Try parsing with dateutil
            date_patterns = [
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?(,\s*\d{4})?\b',
                r'\b\d{1,2}(st|nd|rd|th)?\s+(of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(,\s*\d{4})?\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
            ]
            for pattern in date_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        clean_match = re.sub(r'(st|nd|rd|th)', '', match.group(0))
                        parsed_dt = date_parser.parse(clean_match, default=now)
                        has_date = True
                        iso_date = parsed_dt.strftime("%Y-%m-%d")
                        display_date = parsed_dt.strftime("%A, %b %d, %Y")
                        break
                    except Exception:
                        pass

        # Time parsing
        has_time = False
        iso_time = ""
        display_time = ""

        time_patterns = [
            r'\b\d{1,2}(:\d{2})?\s*(am|pm)\b',
            r'\b\d{1,2}:\d{2}\b'
        ]

        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                time_str = match.group(0)
                try:
                    parsed_t = date_parser.parse(time_str).time()
                    has_time = True
                    iso_time = parsed_t.strftime("%H:%M:%S")
                    display_time = parsed_t.strftime("%I:%M %p").lstrip("0")
                    break
                except Exception:
                    pass

        return {
            "has_date": has_date,
            "has_time": has_time,
            "display_date": display_date or iso_date,
            "display_time": display_time or iso_time,
            "iso_date": iso_date,
            "iso_time": iso_time
        }

    def generate_calendly_links(self, iso_date: str, iso_time: str) -> Tuple[str, str]:
        """
        Generates:
        1. Pre-selected direct slot link (e.g. https://calendly.com/neerajjawahirani/30min/2026-09-23T10:30:00-04:00?month=2026-09&date=2026-09-23)
        2. Standard scheduling link (e.g. https://calendly.com/neerajjawahirani/30min)
        """
        base_url = self.calendly_url
        if not iso_date or not iso_time:
            return base_url, base_url

        month_str = iso_date[:7] # YYYY-MM
        slot_iso = f"{iso_date}T{iso_time}{self.tz_offset}"
        direct_slot_url = f"{base_url}/{slot_iso}?month={month_str}&date={iso_date}"
        return direct_slot_url, base_url

    def process_booking_request(self, query: str, history: Optional[list] = None, llm_client: Any = None) -> Tuple[bool, str]:
        """
        Main entry point to process booking intent.
        Returns:
            (is_handled: bool, response_message: str)
        """
        if not self.is_booking_request(query, history):
            return False, ""

        dt_info = self.extract_datetime_details(query, history, llm_client)

        has_date = dt_info["has_date"]
        has_time = dt_info["has_time"]
        display_date = dt_info["display_date"] or dt_info["iso_date"]
        display_time = dt_info["display_time"] or dt_info["iso_time"]
        iso_date = dt_info["iso_date"]
        iso_time = dt_info["iso_time"]

        # Case 1: Neither date nor time provided
        if not has_date and not has_time:
            return True, (
                "I'd be happy to help you schedule an appointment.\n\n"
                "Please specify your preferred **date** and **time** (for example: *'Tomorrow at 10:30 AM'* or *'Sept 23 at 3:00 PM'*)."
            )

        # Case 2: Date provided, but time missing
        if has_date and not has_time:
            return True, (
                f"Got it! You'd like to book on **{display_date}**.\n\n"
                "What **time** works best for you? (e.g., *'10:30 AM'* or *'2:30 PM'*)"
            )

        # Case 3: Time provided, but date missing
        if not has_date and has_time:
            return True, (
                f"Got it! You'd like a slot at **{display_time}**.\n\n"
                "What **date** would you like to schedule this for? (e.g., *'Tomorrow'* or *'Sept 23'*)"
            )

        # Case 4: Both date and time provided -> Generate Direct pre-selected slot link + Standard link!
        direct_url, standard_url = self.generate_calendly_links(iso_date, iso_time)

        response_msg = (
            f"**Great! Your appointment details are set.**\n\n"
            f"- **Date:** {display_date}\n"
            f"- **Time:** {display_time}\n\n"
            f"**[Click here to confirm this pre-selected slot on Calendly]({direct_url})**\n\n"
            f"--- \n\n"
            f"*Prefer to choose a different time or schedule on your own?*\n"
            f"**[Open Standard Calendly Calendar]({standard_url})**"
        )

        return True, response_msg


# Module-level convenience function
_handler = CalendlyBookingHandler()

def handle_calendly_booking(query: str, history: Optional[list] = None, llm_client: Any = None) -> Tuple[bool, str]:
    """
    Convenience function to handle appointment booking queries.
    """
    return _handler.process_booking_request(query, history, llm_client)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Testing Updated Calendly Booking Handler with Direct Links...")
    handler = CalendlyBookingHandler()

    test_queries = [
        ("I want to book an appointment", []),
        ("Can we meet on 2026-09-23?", []),
        ("At 10:30 AM", [{"user": "Can we meet on 2026-09-23?", "assistant": "What time?"}]),
        ("Schedule a meeting for tomorrow at 3 PM", []),
        ("What are Neeraj's top skills?", [])
    ]

    for q, hist in test_queries:
        print(f"\nUser: '{q}'")
        handled, msg = handler.process_booking_request(q, hist)
        if handled:
            print(f"Assistant Output:\n{msg}")
        else:
            print("Assistant: [Not a booking request, proceeding to RAG]")
