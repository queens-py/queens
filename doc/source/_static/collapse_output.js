document.addEventListener("DOMContentLoaded", () => {
    // Function to create and append a toggle button to a div
    const addToggleButton = (div) => {
        const button = document.createElement("button");
        button.textContent = "Show Output »";
        styleButton(button);
        initializeDivStyle(div);
        addClickEvent(button, div);
        insertButtonAboveDiv(div, button);
    };

    // Styling the toggle button
    const styleButton = (button) => {
        Object.assign(button.style, {
            backgroundColor: "#F5F5F5",
            color: "#333333",
            border: "1px solid #DDD",
            padding: "2px 5px",
            marginTop: "5px",
            cursor: "pointer",
            borderRadius: "3px",
            fontSize: "0.9em",
            width: "100%",
            maxWidth: "100px",
            transition: "max-width 0.5s ease, background-color 1s ease"
        });
    };

    // Initialize div style for hiding
    const initializeDivStyle = (div) => {
        Object.assign(div.style, {
            opacity: '0',
            maxHeight: '0px',
            overflow: 'hidden',
            transition: 'opacity 0.2s ease, max-height 0.6s ease'
        });
    };

    // Handle click event for the toggle button
    const addClickEvent = (button, div) => {
        button.onclick = () => {
            if (div.style.maxHeight === '0px') {
                Object.assign(div.style, {
                    opacity: '1',
                    maxHeight: '50000px',
                });
                button.textContent = "»» Hide Output ««";
                button.style.maxWidth = "100%";
            } else {
                Object.assign(div.style, {
                    opacity: '0',
                    maxHeight: '0px'
                });
                button.textContent = "Show Output »";
                button.style.maxWidth = "100px";
            }
        };
    };

    // Insert the toggle button above the div
    const insertButtonAboveDiv = (div, button) => {
        const hr = document.createElement("hr");
        div.parentNode.insertBefore(hr, div);
        div.parentNode.insertBefore(button, div);
    };

    // Select and apply toggle functionality to code and output divs
    const codeDivs = document.querySelectorAll('.nboutput.docutils.container');
    codeDivs.forEach(addToggleButton);

    // Hide cell numbers (which don't play nice with button layout)
    const outputDivs = document.querySelectorAll('.prompt');
    outputDivs.forEach(div => div.style.display = 'none');
});
