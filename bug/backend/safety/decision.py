class DecisionEngine:

    def decide(self, risk):

        if risk == "CRITICAL":
            return "BRAKE"

        if risk == "DANGER":
            return "BRAKE"

        if risk == "WARNING":
            return "WARNING"

        return "CONTINUE"