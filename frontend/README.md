# Raptor Service Frontend - Atomic Design Structure

## 📁 Project Structure

The frontend project has been refactored to follow **Atomic Design principles** for better maintainability, reusability, and scalability.

```
src/
├── components/
│   ├── atoms/                  # Basic building blocks
│   │   ├── Logo/
│   │   ├── StatCard/
│   │   ├── NavItem/
│   │   ├── IconBox/
│   │   └── index.ts
│   ├── molecules/              # Combinations of atoms
│   │   ├── Navigation/
│   │   ├── FeatureCard/
│   │   ├── TestimonialCard/
│   │   ├── StatsGrid/
│   │   └── index.ts
│   ├── organisms/              # Complex components
│   │   ├── Header/
│   │   ├── HeroSection/
│   │   ├── FeaturesSection/
│   │   ├── TestimonialsSection/
│   │   ├── CTASection/
│   │   ├── Footer/
│   │   └── index.ts
│   ├── templates/              # Page layouts
│   │   ├── LandingPageTemplate/
│   │   └── index.ts
│   └── index.ts
├── pages/                      # Complete pages
│   ├── HomePage.tsx
│   └── index.ts
├── constants/                  # Data constants
│   ├── heroData.ts
│   ├── featuresData.ts
│   ├── testimonialsData.ts
│   └── index.ts
├── hooks/                      # Custom React hooks
├── utils/                      # Utility functions
├── App.tsx                     # Main application
└── main.tsx                    # Application entry point
```

## 🧩 Atomic Design Methodology

### Atoms

**Basic building blocks** - Smallest components that can't be broken down further:

- `Logo` - Brand logo with optional text
- `StatCard` - Display statistics with value and label
- `NavItem` - Individual navigation link
- `IconBox` - Wrapper for icons with styling variants

### Molecules

**Combinations of atoms** - Groups of atoms bonded together:

- `Navigation` - Group of navigation items
- `FeatureCard` - Feature display with icon, title, and description
- `TestimonialCard` - Customer testimonial with rating and author info
- `StatsGrid` - Grid layout for statistics

### Organisms

**Complex components** - Groups of molecules joined together:

- `Header` - Site header with logo, navigation, and CTA
- `HeroSection` - Main hero area with headline, description, and actions
- `FeaturesSection` - Features showcase section
- `TestimonialsSection` - Customer testimonials section
- `CTASection` - Call-to-action section
- `Footer` - Site footer with links and info

### Templates

**Page layouts** - Collections of organisms forming page structures:

- `LandingPageTemplate` - Layout for landing pages with header and footer

### Pages

**Complete pages** - Specific instances of templates with real content:

- `HomePage` - Main landing page

## 🚀 Development Guidelines

### Creating New Components

1. **Identify the component level** (atom, molecule, organism)
2. **Create folder structure**:
   ```
   ComponentName/
   ├── ComponentName.tsx
   └── index.ts
   ```
3. **Export from parent index.ts**
4. **Use TypeScript interfaces** for props
5. **Follow naming conventions** (PascalCase for components)

### Component Props Pattern

```typescript
interface ComponentNameProps {
  // Required props first
  title: string;

  // Optional props with defaults
  size?: "sm" | "md" | "lg";
  variant?: "primary" | "secondary";

  // Always include className for styling flexibility
  className?: string;

  // Event handlers
  onClick?: () => void;
}
```

### Data Management

- Keep data in `constants/` directory
- Export from main constants index
- Use TypeScript interfaces for data structures

## 📦 Package Management

This project uses **pnpm** with **corepack** for package management:

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Run linting
pnpm lint
```

## 🎨 Styling

- **TailwindCSS** for utility-first styling
- **Radix UI Themes** for design system components
- **HeroUI React** for enhanced UI components
- **Responsive design** with mobile-first approach

## 🔧 Benefits of This Structure

1. **Reusability** - Components can be easily reused across pages
2. **Maintainability** - Clear separation of concerns
3. **Scalability** - Easy to add new components and features
4. **Testing** - Components can be tested in isolation
5. **Collaboration** - Clear structure for team development
6. **Performance** - Better tree-shaking and code splitting potential

## 📋 Available Scripts

- `pnpm dev` - Start development server
- `pnpm build` - Build for production
- `pnpm lint` - Run ESLint
- `pnpm preview` - Preview production build

## 🎯 Next Steps

1. Add routing with React Router or Next.js
2. Implement state management (Zustand, Redux Toolkit)
3. Add unit tests for components
4. Set up Storybook for component documentation
5. Add accessibility improvements
6. Implement error boundaries
