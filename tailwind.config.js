module.exports = {
    theme: {
      extend: {
        keyframes: {
          scale: {
            "0%, 100%": { transform: "scale(1)" },
            "50%": { transform: "scale(1.1)" },
          },
        },
        animation: {
          scale: "scale 1s infinite ease-in-out",
        },
      },
    },
  };
  