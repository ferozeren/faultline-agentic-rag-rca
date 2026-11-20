import { createContext, useContext, useMemo, useState } from "react";

const InvestigationContext = createContext(null);

export function InvestigationProvider({ children }) {
  const [latestInvestigation, setLatestInvestigation] = useState(null);
  const value = useMemo(
    () => ({ latestInvestigation, saveInvestigation: setLatestInvestigation }),
    [latestInvestigation],
  );

  return (
    <InvestigationContext.Provider value={value}>
      {children}
    </InvestigationContext.Provider>
  );
}

export function useLatestInvestigation() {
  const context = useContext(InvestigationContext);
  if (!context) {
    throw new Error("useLatestInvestigation must be used within InvestigationProvider.");
  }
  return context;
}
