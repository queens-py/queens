document.addEventListener("DOMContentLoaded", function () {
    const outputs = document.querySelectorAll('.nboutput.docutils.container');
    let prev = null;

    outputs.forEach(output => {
        if (
            prev &&
            // no intervening non-output node
            prev.nextElementSibling === output &&
            // same type of prompt (empty) and no distinguishing marks
            prev.querySelector('.output_area') &&
            output.querySelector('.output_area')
        ) {
            // Move inner content of current into previous
            const content = output.querySelector('.output_area');
            if (content) {
                prev.appendChild(content);
                output.remove();  // remove the now-empty container
            }
        } else {
            prev = output;
        }
    });
});


// document.addEventListener("DOMContentLoaded", function () {
//     const outputs = document.querySelectorAll('.nboutput.docutils.container');
//     let prev = null;

//     outputs.forEach(output => {
//         if (
//             prev &&
//             prev.nextElementSibling === output &&
//             prev.querySelector('.output_area') &&
//             output.querySelector('.output_area')
//         ) {
//             // Move the entire .output_area block from output to prev
//             const content = output.querySelector('.output_area');
//             if (content) {
//                 const clone = content.cloneNode(true);
//                 // Insert a vertical spacer to enforce block stacking
//                 const spacer = document.createElement('div');
//                 spacer.style.height = '0.5em';
//                 prev.appendChild(spacer);
//                 prev.appendChild(clone);
//                 output.remove();
//             }
//         } else {
//             prev = output;
//         }
//     });
// });

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
            maxWidth: "150px",
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
                    display: 'block',
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
                button.style.maxWidth = "150px";
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
    const codeDivs = document.querySelectorAll('.nboutput');
    codeDivs.forEach(addToggleButton);

    // Hide cell numbers (which don't play nice with button layout)
    const outputDivs = document.querySelectorAll('.prompt');
    outputDivs.forEach(div => div.style.display = 'none');
});
