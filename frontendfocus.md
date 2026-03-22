claude "Read SPEC.md sections 8 and 10. Build a hero landing page 
component at frontend/components/HeroSection.tsx. 

Visual references I'm going for: stripesessions.com (Three.js 
particle globe with shader glow), reflect.app (glass cards with 
backdrop blur, grain texture), cash.app (bold CTA, clean dark 
sections).

Requirements:
- Three.js particle globe on the right side, 4000+ dots, orange/amber, 
  auto-rotating, with bright hotspots at the 6 metro areas from the spec
- Atmosphere glow shader behind the globe
- Orbital arc rings around the globe
- Curved bezier paths connecting metro hotspots
- 3 floating glass cards overlaid on the globe showing:
  Card 1: Washington DC, 3.2% default, 'FICO OVERESTIMATES RISK' green badge
  Card 2: San Francisco, 'Running 5,000 paths', mini fan chart SVG
  Card 3: Gas spike stress test, 8.1% default, 'FICO UNDERESTIMATES RISK' red badge
- Headline: 'Simulate income, predict risk.' orange, 'Beyond FICO.' white
- Subtitle label: 'MONTE CARLO CREDIT INTELLIGENCE'
- Background: near-black with CSS grain overlay and radial orange glow
- Blue CTA button: 'Launch Simulator'
- Bottom stat bar: 5,000 paths / 36% CV / 6 metros / 36 months
- Source badges: JPMorgan Chase Institute, Federal Reserve, Gridwise, BLS
- Font: Syne for display, DM Sans for body
- Staggered entrance animations on everything
- Dark mode only, no mobile needed"