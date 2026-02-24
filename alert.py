import logging


class AlertSystem:
    def __init__(self, email_enabled=False):
        self.email_enabled = email_enabled

    def send_alert(self, message):
        logging.warning(message)