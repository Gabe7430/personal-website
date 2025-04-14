module.exports = {
    theme: {
      extend: {
        keyframes: {
          scale: {
            "0%, 100%": { transform: "scale(1)" },
            "50%": { transform: "scale(1.1)" },
          },
          'pulse-slow': {
            '0%, 100%': { opacity: '1' },
            '50%': { opacity: '0.85' },
          },
          fadeIn: {
            '0%': { opacity: '0.4', transform: 'translateY(10px)' },
            '100%': { opacity: '1', transform: 'translateY(0)' },
          },
        },
        animation: {
          scale: "scale 1s infinite ease-in-out",
          'pulse-slow': 'pulse-slow 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          'fadeIn': 'fadeIn 0.8s ease-out forwards',
        },
      },
    },
  };
  