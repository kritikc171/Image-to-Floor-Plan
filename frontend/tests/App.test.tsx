import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "../src/App";

describe("App", () => {
  it("renders the product name", () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { name: /Floor Plan Generator/i, level: 1 }),
    ).toBeInTheDocument();
  });

  it("shows the approximation disclaimer", () => {
    render(<App />);
    expect(screen.getByText(/Approximate results/i)).toBeInTheDocument();
  });
});
