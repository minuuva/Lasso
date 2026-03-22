# Lasso Frontend

Next.js frontend for the Lasso Monte Carlo credit risk assessment platform.

## Prerequisites

- Node.js 18+
- npm (comes with Node.js)

## Getting Started

1. Install dependencies:

```bash
npm install
```

2. Create a `.env.local` file with your API key:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

3. Run the development server:

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) with your browser.

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run typecheck` - Run TypeScript type checking

## Tech Stack

- Next.js 16
- React 19
- Tailwind CSS v4
- GSAP for animations
- Three.js / React Three Fiber for 3D globe
- Recharts for data visualization
