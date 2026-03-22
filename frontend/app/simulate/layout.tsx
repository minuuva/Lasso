export default function SimulateLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // This layout intentionally does NOT include SmoothScrollProvider
  // because the simulate page is a fixed-height app interface
  return <>{children}</>;
}
