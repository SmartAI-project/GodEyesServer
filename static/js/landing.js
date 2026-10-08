const navbar = document.getElementById("navbar");

let lastScrollY = window.scrollY;
let scrollDirection = "down";
let ticking = false;

function updateScrollDirection() {
    const currentY = window.scrollY;

    if (currentY > lastScrollY + 2) {
        scrollDirection = "down";
    } else if (currentY < lastScrollY - 2) {
        scrollDirection = "up";
    }

    lastScrollY = currentY;

    if (navbar) {
        navbar.classList.toggle("scrolled", currentY > 35);
    }

    ticking = false;
}

window.addEventListener("scroll", () => {
    if (!ticking) {
        window.requestAnimationFrame(updateScrollDirection);
        ticking = true;
    }
}, { passive: true });

const revealItems = document.querySelectorAll(".reveal");

const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        const element = entry.target;

        if (entry.isIntersecting) {
            element.classList.remove("reveal-up", "reveal-down", "reveal-left", "reveal-right");

            if (scrollDirection === "up") {
                element.classList.add(
                    element.dataset.upDirection || "reveal-down"
                );
            } else {
                element.classList.add(
                    element.dataset.downDirection || "reveal-up"
                );
            }

            requestAnimationFrame(() => {
                element.classList.add("visible");
            });
        } else {
            element.classList.remove("visible");
        }
    });
}, {
    threshold: 0.12,
    rootMargin: "-5% 0px -8% 0px"
});

revealItems.forEach((element) => {
    const original = [...element.classList];

    if (original.includes("reveal-left")) {
        element.dataset.downDirection = "reveal-left";
        element.dataset.upDirection = "reveal-right";
    } else if (original.includes("reveal-right")) {
        element.dataset.downDirection = "reveal-right";
        element.dataset.upDirection = "reveal-left";
    } else {
        element.dataset.downDirection = "reveal-up";
        element.dataset.upDirection = "reveal-down";
    }

    revealObserver.observe(element);
});

document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (event) => {
        const targetId = link.getAttribute("href");
        if (!targetId || targetId === "#") return;

        const target = document.querySelector(targetId);
        if (!target) return;

        event.preventDefault();

        target.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    });
});
