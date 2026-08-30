
// ================= NAVIGATION MENU =================

const menu = document.querySelector('.menu');
const links = document.querySelector('.nav-links');

menu?.addEventListener('click', () => {
  links.classList.toggle('open');
});

document.querySelectorAll('.nav-links a').forEach(link => {
  link.addEventListener('click', () => {
    links.classList.remove('open');
  });
});


// ================= SCROLL REVEAL =================

const observer = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  },
  {
    threshold: 0.08
  }
);

document.querySelectorAll('.reveal').forEach(element => {
  observer.observe(element);
});


// ================= PROJECT FILTERS =================

const buttons = document.querySelectorAll('.filters button');
const cards = document.querySelectorAll('.project');

buttons.forEach(button => {

  button.addEventListener('click', () => {

    // Remove active state
    buttons.forEach(btn => {
      btn.classList.remove('active');
    });

    // Add active state
    button.classList.add('active');

    const filter = button.dataset.filter;

    // Show / hide projects
    cards.forEach(card => {

      if (
        filter === 'All' ||
        card.dataset.category === filter
      ) {
        card.style.display = 'flex';
      } else {
        card.style.display = 'none';
      }

    });

  });

});


// ================= FORMSPREE CONTACT FORM =================

const form = document.getElementById('contactForm');

form?.addEventListener('submit', () => {

  const status = document.getElementById('formStatus');

  if (status) {
    status.textContent = 'Sending message…';
  }

});

